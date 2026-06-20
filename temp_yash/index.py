

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. CORE LOGIC (UNCHANGED BEHAVIOR)
# ==========================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

class FakeNewsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Fake News Detector")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e2e") # Modern dark background
        
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        
        self.setup_gui()
        self.initialize_ml()

    def setup_gui(self):
        # Header Section
        header = tk.Label(self.root, text="🛡️ FAKE NEWS DETECTOR", font=("Helvetica", 22, "bold"), 
                         fg="#cba6f7", bg="#1e1e2e", pady=20)
        header.pack()

        # Input Section
        input_label = tk.Label(self.root, text="Paste the news article below:", font=("Helvetica", 12), 
                              fg="#a6adc8", bg="#1e1e2e")
        input_label.pack(anchor="w", padx=50)

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=70, height=12, 
                                                 font=("Helvetica", 11), bg="#313244", fg="#cdd6f4",
                                                 insertbackground="white", borderwidth=0)
        self.text_area.pack(pady=10, padx=50)

        # Buttons Section
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(pady=20)

        self.check_btn = tk.Button(btn_frame, text="Analyze News", command=self.predict_news, 
                                  font=("Helvetica", 12, "bold"), bg="#a6e3a1", fg="#11111b",
                                  padx=20, pady=10, activebackground="#94e2d5", cursor="hand2")
        self.check_btn.grid(row=0, column=0, padx=10)

        clear_btn = tk.Button(btn_frame, text="Clear", command=lambda: self.text_area.delete('1.0', tk.END), 
                             font=("Helvetica", 12), bg="#f38ba8", fg="#11111b",
                             padx=20, pady=10, activebackground="#eba0ac", cursor="hand2")
        clear_btn.grid(row=0, column=1, padx=10)

        # Result Section
        self.result_label = tk.Label(self.root, text="System Ready", font=("Helvetica", 16, "bold"), 
                                    fg="#89b4fa", bg="#1e1e2e", pady=20)
        self.result_label.pack()

        # Footer / Accuracy
        self.status_bar = tk.Label(self.root, text="Initializing AI...", bd=1, relief=tk.SUNKEN, 
                                  anchor="w", bg="#181825", fg="#6c7086", font=("Helvetica", 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def initialize_ml(self):
        """Loads data and trains the model exactly as the original logic did."""
        try:
            df_true = pd.read_csv("true.csv")
            df_fake = pd.read_csv("fake.csv")
            
            df_true["label"] = 1
            df_fake["label"] = 0
            df = pd.concat([df_true, df_fake], axis=0).sample(frac=1).reset_index(drop=True)
            df = df[['text', 'label']]

            df["text"] = df["text"].apply(clean_text)
            
            x_train, x_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3)
            
            xv_train = self.vectorizer.fit_transform(x_train)
            xv_test = self.vectorizer.transform(x_test)
            
            self.model.fit(xv_train, y_train)
            accuracy = self.model.score(xv_test, y_test)
            
            self.status_bar.config(text=f"✅ AI Model Active | Accuracy: {accuracy*100:.2f}%")
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Make sure true.csv and fake.csv are in the folder!\n\n{e}")
            self.root.destroy()

    def predict_news(self):
        input_text = self.text_area.get("1.0", "end-1c")
        if not input_text.strip():
            messagebox.showwarning("Empty Input", "Please paste some news text first!")
            return

        cleaned_input = clean_text(input_text)
        vector_input = self.vectorizer.transform([cleaned_input])
        prediction = self.model.predict(vector_input)

        if prediction[0] == 1:
            self.result_label.config(text="✅ RESULT: GENUINE NEWS", fg="#a6e3a1")
        else:
            self.result_label.config(text="🚩 RESULT: FAKE NEWS", fg="#f38ba8")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsApp(root)
    root.mainloop()
