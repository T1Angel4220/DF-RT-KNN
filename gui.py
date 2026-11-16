"""
DF-RT-KNN: Interfaz Gráfica Interactiva para Clasificación de Dígitos MNIST
Permite explorar muestras del conjunto de prueba, dibujar dígitos a mano alzada
y comparar las predicciones en tiempo real de Árbol de Decisión, Random Forest y k-NN.
"""

import sys
import time
import re
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# Parches de compatibilidad para Python 3.14+ y scikit-learn
_orig_re_search = re.search
def _safe_re_search(pattern, string, flags=0):
    if not isinstance(string, (str, bytes, bytearray)):
        string = str(string)
    return _orig_re_search(pattern, string, flags)
re.search = _safe_re_search

try:
    from pandas.core.arrays.categorical import Categorical
    _orig_rename_categories = Categorical.rename_categories
    def _safe_rename_categories(self, new_categories, *args, **kwargs):
        if callable(new_categories):
            orig_func = new_categories
            def _wrapped(val):
                try:
                    return orig_func(val)
                except Exception:
                    return str(val)
            return _orig_rename_categories(self, _wrapped, *args, **kwargs)
        return _orig_rename_categories(self, new_categories, *args, **kwargs)
    Categorical.rename_categories = _safe_rename_categories
except Exception:
    pass

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class MNISTClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Classifier Studio — Árbol de Decisión | Random Forest | k-NN")
        self.root.geometry("1020x680")
        self.root.minsize(960, 620)

        # Configuración de estilo
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Variables de estado
        self.canvas_size = 280
        self.pixel_size = 10  # 28x28 -> 280x280
        self.current_idx = 0
        self.draw_mode = False
        self.custom_digit_grid = np.zeros((28, 28), dtype=np.float32)

        # Contenedores de datos y modelos
        self.X_ent = None
        self.y_ent = None
        self.X_pru = None
        self.y_pru = None
        self.modelos = {}
        self.tiempos_entrenamiento = {}
        self.predicciones_pru = {}
        self.indices_fallos = []

        self._crear_interfaz_carga()
        self.root.after(100, self._inicializar_datos_y_modelos)

    def _crear_interfaz_carga(self):
        self.frame_loading = ttk.Frame(self.root, padding=20)
        self.frame_loading.pack(expand=True, fill="both")

        lbl_title = ttk.Label(
            self.frame_loading,
            text="MNIST Classifier Studio",
            font=("Segoe UI", 18, "bold")
        )
        lbl_title.pack(pady=(60, 10))

        self.lbl_status = ttk.Label(
            self.frame_loading,
            text="Cargando dataset MNIST y ajustando modelos (Decision Tree, RF, k-NN)...",
            font=("Segoe UI", 11)
        )
        self.lbl_status.pack(pady=10)

        self.progress = ttk.Progressbar(self.frame_loading, mode="indeterminate", length=400)
        self.progress.pack(pady=20)
        self.progress.start(10)

    def _inicializar_datos_y_modelos(self):
        try:
            X, y = fetch_openml(data_id=554, as_frame=False, return_X_y=True)
            X = X.astype("float32") / 255.0
            y = y.astype("int64")

            X_base_ent, y_base_ent = X[:60000], y[:60000]
            X_base_pru, y_base_pru = X[60000:], y[60000:]

            self.X_ent, _, self.y_ent, _ = train_test_split(
                X_base_ent, y_base_ent, train_size=10000, stratify=y_base_ent, random_state=42
            )
            self.X_pru, _, self.y_pru, _ = train_test_split(
                X_base_pru, y_base_pru, train_size=2000, stratify=y_base_pru, random_state=42
            )

            self.modelos = {
                "Árbol de Decisión": DecisionTreeClassifier(max_depth=15, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
                "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform", n_jobs=-1)
            }

            for nombre, modelo in self.modelos.items():
                t0 = time.time()
                modelo.fit(self.X_ent, self.y_ent)
                self.tiempos_entrenamiento[nombre] = time.time() - t0
                self.predicciones_pru[nombre] = modelo.predict(self.X_pru)

            for i in range(len(self.y_pru)):
                fallo = any(self.predicciones_pru[n][i] != self.y_pru[i] for n in self.modelos)
                if fallo:
                    self.indices_fallos.append(i)

            self.progress.stop()
            self.frame_loading.destroy()
            self._construir_interfaz_principal()
            self._cargar_muestra_prueba(0)

        except Exception as e:
            self.progress.stop()
            messagebox.showerror("Error al inicializar", f"No se pudo inicializar el modelo:\n{e}")

    def _construir_interfaz_principal(self):
        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(expand=True, fill="both")

        # Panel Izquierdo: Visualizador y Lienzo
        panel_izq = ttk.LabelFrame(main_container, text=" Visualizador de Dígitos (28x28) ", padding=10)
        panel_izq.pack(side="left", fill="both", expand=False, padx=(0, 10))

        self.canvas = tk.Canvas(
            panel_izq,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#555"
        )
        self.canvas.pack(pady=(0, 10))
        self.canvas.bind("<B1-Motion>", self._dibujar_en_lienzo)
        self.canvas.bind("<Button-1>", self._dibujar_en_lienzo)

        # Controles de Muestras
        frame_nav = ttk.Frame(panel_izq)
        frame_nav.pack(fill="x", pady=4)

        btn_prev = ttk.Button(frame_nav, text="◀ Anterior", width=10, command=self._muestra_anterior)
        btn_prev.pack(side="left", padx=2)

        btn_rand = ttk.Button(frame_nav, text="🎲 Aleatoria", width=10, command=self._muestra_aleatoria)
        btn_rand.pack(side="left", padx=2)

        btn_next = ttk.Button(frame_nav, text="Siguiente ▶", width=10, command=self._muestra_siguiente)
        btn_next.pack(side="left", padx=2)

        frame_special = ttk.Frame(panel_izq)
        frame_special.pack(fill="x", pady=4)

        btn_fail = ttk.Button(
            frame_special,
            text="⚠️ Ver Caso Mal Clasificado",
            command=self._ver_caso_fallido
        )
        btn_fail.pack(fill="x", pady=2)

        frame_draw = ttk.LabelFrame(panel_izq, text=" Dibujo a Mano Alzada ", padding=6)
        frame_draw.pack(fill="x", pady=(10, 0))

        btn_clear = ttk.Button(frame_draw, text="🧹 Limpiar Lienzo", command=self._limpiar_lienzo)
        btn_clear.pack(side="left", expand=True, fill="x", padx=2)

        btn_classify_draw = ttk.Button(frame_draw, text="⚡ Clasificar Trazo", command=self._clasificar_trazo)
        btn_classify_draw.pack(side="left", expand=True, fill="x", padx=2)

        # Panel Derecho: Métricas, Predicciones y Comparativa
        panel_der = ttk.Frame(main_container)
        panel_der.pack(side="right", fill="both", expand=True)

        # Encabezado de la muestra actual
        frame_info = ttk.LabelFrame(panel_der, text=" Información de la Muestra ", padding=10)
        frame_info.pack(fill="x", pady=(0, 10))

        self.lbl_muestra_info = ttk.Label(
            frame_info,
            text="Muestra: Conjunto de Prueba #0 | Etiqueta Real: 7",
            font=("Segoe UI", 12, "bold")
        )
        self.lbl_muestra_info.pack(anchor="w")

        # Botón principal de clasificación
        btn_eval_all = ttk.Button(
            panel_der,
            text="🔍 Evaluar con los Tres Modelos Simultáneamente",
            command=self._evaluar_muestra_actual
        )
        btn_eval_all.pack(fill="x", pady=(0, 10))

        # Tabla comparativa de resultados individuales
        frame_cards = ttk.LabelFrame(panel_der, text=" Resultados de Predicción por Modelo ", padding=10)
        frame_cards.pack(fill="x", pady=(0, 10))

        self.card_labels = {}
        for idx, nombre in enumerate(self.modelos.keys()):
            row_frame = ttk.Frame(frame_cards, padding=4)
            row_frame.pack(fill="x", pady=3)

            lbl_name = ttk.Label(row_frame, text=f"• {nombre}:", font=("Segoe UI", 10, "bold"), width=22)
            lbl_name.pack(side="left")

            lbl_pred = ttk.Label(row_frame, text="Predicción: -", font=("Segoe UI", 10), width=18)
            lbl_pred.pack(side="left")

            lbl_status = ttk.Label(row_frame, text="Estado: -", font=("Segoe UI", 10), width=15)
            lbl_status.pack(side="left")

            self.card_labels[nombre] = {
                "pred": lbl_pred,
                "status": lbl_status
            }

        # Resumen Global de Rendimiento (Benchmark)
        frame_bench = ttk.LabelFrame(panel_der, text=" Tabla Oficial de Rendimiento (Test Set n=2,000) ", padding=10)
        frame_bench.pack(fill="both", expand=True)

        columns = ("Modelo", "Acc. Ent (%)", "Error Ent (%)", "Acc. Pru (%)", "Error Pru (%)", "Brecha (p.p.)", "T. Fit (s)")
        self.tree_bench = ttk.Treeview(frame_bench, columns=columns, show="headings", height=4)
        
        for col in columns:
            self.tree_bench.heading(col, text=col)
            self.tree_bench.column(col, anchor="center", width=95)
        self.tree_bench.column("Modelo", anchor="w", width=140)

        # Llenar tabla de benchmark
        for nombre, modelo in self.modelos.items():
            acc_ent = accuracy_score(self.y_ent, modelo.predict(self.X_ent)) * 100.0
            acc_pru = accuracy_score(self.y_pru, self.predicciones_pru[nombre]) * 100.0
            err_ent = 100.0 - acc_ent
            err_pru = 100.0 - acc_pru
            brecha = acc_ent - acc_pru
            t_fit = self.tiempos_entrenamiento.get(nombre, 0.0)

            self.tree_bench.insert("", "end", values=(
                nombre,
                f"{acc_ent:.2f}",
                f"{err_ent:.2f}",
                f"{acc_pru:.2f}",
                f"{err_pru:.2f}",
                f"{brecha:.2f}",
                f"{t_fit:.2f}"
            ))

        self.tree_bench.pack(fill="both", expand=True, pady=5)

    def _renderizar_imagen(self, img_array_28x28):
        self.canvas.delete("all")
        for r in range(28):
            for c in range(28):
                val = int(np.clip(img_array_28x28[r, c] * 255.0, 0, 255))
                hex_color = f"#{val:02x}{val:02x}{val:02x}"
                x0 = c * self.pixel_size
                y0 = r * self.pixel_size
                x1 = x0 + self.pixel_size
                y1 = y0 + self.pixel_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=hex_color, outline="")

    def _cargar_muestra_prueba(self, idx):
        self.draw_mode = False
        self.current_idx = idx % len(self.X_pru)
        img_28x28 = self.X_pru[self.current_idx].reshape(28, 28)
        self._renderizar_imagen(img_28x28)

        real_label = self.y_pru[self.current_idx]
        self.lbl_muestra_info.config(
            text=f"Muestra: Conjunto de Prueba #{self.current_idx} | Etiqueta Real: {real_label}"
        )
        self._evaluar_muestra_actual()

    def _muestra_siguiente(self):
        self._cargar_muestra_prueba(self.current_idx + 1)

    def _muestra_anterior(self):
        self._cargar_muestra_prueba(self.current_idx - 1)

    def _muestra_aleatoria(self):
        rand_idx = np.random.randint(0, len(self.X_pru))
        self._cargar_muestra_prueba(rand_idx)

    def _ver_caso_fallido(self):
        if self.indices_fallos:
            f_idx = self.indices_fallos[0]
            self._cargar_muestra_prueba(f_idx)
            # Rotar casos de fallo
            self.indices_fallos.append(self.indices_fallos.pop(0))

    def _dibujar_en_lienzo(self, event):
        self.draw_mode = True
        c = event.x // self.pixel_size
        r = event.y // self.pixel_size

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 28 and 0 <= nc < 28:
                    intensity = 1.0 if (dr == 0 and dc == 0) else 0.5
                    self.custom_digit_grid[nr, nc] = min(1.0, self.custom_digit_grid[nr, nc] + intensity)

        self._renderizar_imagen(self.custom_digit_grid)
        self.lbl_muestra_info.config(
            text="Modo: Trazo a Mano Alzada (Dibujo del Usuario) | Etiqueta Real: Desconocida"
        )

    def _limpiar_lienzo(self):
        self.draw_mode = True
        self.custom_digit_grid = np.zeros((28, 28), dtype=np.float32)
        self.canvas.delete("all")
        self.lbl_muestra_info.config(
            text="Lienzo en blanco — Dibuja un dígito con el ratón"
        )
        for nombre in self.modelos:
            self.card_labels[nombre]["pred"].config(text="Predicción: -")
            self.card_labels[nombre]["status"].config(text="Estado: -", foreground="black")

    def _clasificar_trazo(self):
        self._evaluar_vector(self.custom_digit_grid.flatten(), real_label=None)

    def _evaluar_muestra_actual(self):
        if self.draw_mode:
            self._clasificar_trazo()
        else:
            vector = self.X_pru[self.current_idx]
            real = self.y_pru[self.current_idx]
            self._evaluar_vector(vector, real_label=real)

    def _evaluar_vector(self, vector, real_label=None):
        X_eval = vector.reshape(1, -1)

        for nombre, modelo in self.modelos.items():
            pred = modelo.predict(X_eval)[0]
            self.card_labels[nombre]["pred"].config(text=f"Predicción:  Dígito {pred}")

            if real_label is not None:
                if pred == real_label:
                    self.card_labels[nombre]["status"].config(text="✓ Acertado", foreground="#1b5e20")
                else:
                    self.card_labels[nombre]["status"].config(text="✗ Fallo", foreground="#b71c1c")
            else:
                self.card_labels[nombre]["status"].config(text="Inferencia OK", foreground="#0d47a1")


def main():
    root = tk.Tk()
    app = MNISTClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
