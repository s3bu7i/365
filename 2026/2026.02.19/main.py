import os
import json
import hashlib
from datetime import datetime
from cryptography.fernet import Fernet
import tkinter as tk
from tkinter import messagebox, scrolledtext
import sys

class NotesApp:
    def __init__(self):
        # Proqramın işlədiyi qovluq
        self.app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.notes_dir = os.path.join(self.app_dir, "notes_data")
        self.key_file = os.path.join(self.app_dir, ".secret_key")
        
        os.makedirs(self.notes_dir, exist_ok=True)
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
        
    def _get_or_create_key(self):
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        return key
    
    def get_all_notes(self):
        notes = []
        for filename in os.listdir(self.notes_dir):
            if filename.endswith('.enc'):
                try:
                    filepath = os.path.join(self.notes_dir, filename)
                    with open(filepath, 'rb') as f:
                        encrypted = f.read()
                        decrypted = self.cipher.decrypt(encrypted)
                        note = json.loads(decrypted.decode())
                        notes.append(note)
                except:
                    pass
        return sorted(notes, key=lambda x: x['created'], reverse=True)
    
    def save_note(self, title, content):
        note_id = hashlib.md5(f"{title}{datetime.now()}".encode()).hexdigest()[:8]
        note = {
            'id': note_id,
            'title': title,
            'content': content,
            'created': datetime.now().strftime("%d.%m.%Y %H:%M")
        }
        
        json_data = json.dumps(note, ensure_ascii=False)
        encrypted = self.cipher.encrypt(json_data.encode())
        
        filepath = os.path.join(self.notes_dir, f"{note_id}.enc")
        with open(filepath, 'wb') as f:
            f.write(encrypted)
        
        return note_id
    
    def get_note(self, note_id):
        filepath = os.path.join(self.notes_dir, f"{note_id}.enc")
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            encrypted = f.read()
            decrypted = self.cipher.decrypt(encrypted)
            return json.loads(decrypted.decode())
    
    def delete_note(self, note_id):
        filepath = os.path.join(self.notes_dir, f"{note_id}.enc")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

class RetroNotesGUI:
    def __init__(self, root):
        self.root = root
        self.app = NotesApp()
        self.setup_window()
        self.show_main_page()
        
    def setup_window(self):
        self.root.title("🔒 GİZLİ NOT DƏFTƏRİ v1.0")
        self.root.geometry("700x600")
        self.root.configure(bg='#c0c0c0')
        self.root.resizable(True, True)
        
        # 90s style colors
        self.bg_color = '#c0c0c0'
        self.title_bar_color = '#000080'
        self.text_color = '#000000'
        self.highlight_color = '#ffffe1'
        
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def create_title_bar(self, text):
        title_frame = tk.Frame(self.root, bg=self.title_bar_color, height=30)
        title_frame.pack(fill='x', pady=2, padx=2)
        
        title_label = tk.Label(
            title_frame, 
            text=text, 
            bg=self.title_bar_color, 
            fg='white',
            font=('Arial', 10, 'bold'),
            anchor='w',
            padx=5
        )
        title_label.pack(side='left', fill='both', expand=True)
        
        close_label = tk.Label(
            title_frame,
            text='[_][□][X]',
            bg=self.title_bar_color,
            fg='white',
            font=('Arial', 10)
        )
        close_label.pack(side='right', padx=5)
    
    def create_button(self, parent, text, command, width=15):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Arial', 10),
            width=width,
            bg=self.bg_color,
            relief='raised',
            bd=3,
            cursor='hand2'
        )
        return btn
    
    def show_main_page(self):
        self.clear_window()
        self.create_title_bar("🔒 GİZLİ NOT DƏFTƏRİ v1.0")
        
        # Marquee
        marquee_frame = tk.Frame(self.root, bg='yellow', bd=2, relief='solid')
        marquee_frame.pack(fill='x', padx=15, pady=10)
        
        marquee_label = tk.Label(
            marquee_frame,
            text="🔐 Bütün notlarınız şifrələnmiş şəkildə saxlanılır! 🔐",
            bg='yellow',
            font=('Arial', 9, 'bold')
        )
        marquee_label.pack(pady=5)
        
        # Header
        header_label = tk.Label(
            self.root,
            text="📝 NOTLARIM",
            bg=self.bg_color,
            font=('Arial', 14, 'bold')
        )
        header_label.pack(pady=10)
        
        # New Note Button
        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(pady=5)
        
        new_btn = self.create_button(btn_frame, "➕ YENİ NOT", self.show_new_note_page)
        new_btn.pack()
        
        # Notes List Frame
        list_frame = tk.Frame(self.root, bg='white', bd=2, relief='sunken')
        list_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Listbox
        self.notes_listbox = tk.Listbox(
            list_frame,
            font=('Courier New', 10),
            bg='white',
            selectbackground='#000080',
            selectforeground='white',
            yscrollcommand=scrollbar.set,
            bd=0
        )
        self.notes_listbox.pack(fill='both', expand=True)
        scrollbar.config(command=self.notes_listbox.yview)
        
        # Load notes
        self.notes = self.app.get_all_notes()
        
        if not self.notes:
            self.notes_listbox.insert(tk.END, "Hələ not yoxdur...")
        else:
            for note in self.notes:
                display_text = f"{note['title']:<40} | {note['created']}"
                self.notes_listbox.insert(tk.END, display_text)
        
        # Buttons
        btn_bottom_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_bottom_frame.pack(pady=10)
        
        view_btn = self.create_button(btn_bottom_frame, "👁️ OXUMA", self.view_note, 12)
        view_btn.pack(side='left', padx=5)
        
        delete_btn = self.create_button(btn_bottom_frame, "🗑️ SİL", self.delete_note, 12)
        delete_btn.pack(side='left', padx=5)
        
        # Status Bar
        status_frame = tk.Frame(self.root, bg=self.bg_color, bd=2, relief='groove')
        status_frame.pack(fill='x', padx=2, pady=2)
        
        status_text = f"Cəmi notlar: {len(self.notes)} | Tarix: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        status_label = tk.Label(
            status_frame,
            text=status_text,
            bg=self.bg_color,
            font=('Arial', 8),
            anchor='w',
            padx=5
        )
        status_label.pack(fill='x')
    
    def show_new_note_page(self):
        self.clear_window()
        self.create_title_bar("📝 YENİ NOT YAZ")
        
        # Content Frame
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Title
        title_label = tk.Label(
            content_frame,
            text="📌 Başlıq:",
            bg=self.bg_color,
            font=('Arial', 10, 'bold'),
            anchor='w'
        )
        title_label.pack(fill='x', pady=(0, 5))
        
        self.title_entry = tk.Entry(
            content_frame,
            font=('Arial', 11),
            bd=2,
            relief='sunken'
        )
        self.title_entry.pack(fill='x', pady=(0, 15))
        self.title_entry.focus()
        
        # Content
        content_label = tk.Label(
            content_frame,
            text="📄 Məzmun:",
            bg=self.bg_color,
            font=('Arial', 10, 'bold'),
            anchor='w'
        )
        content_label.pack(fill='x', pady=(0, 5))
        
        self.content_text = scrolledtext.ScrolledText(
            content_frame,
            font=('Arial', 11),
            bd=2,
            relief='sunken',
            wrap='word',
            height=15
        )
        self.content_text.pack(fill='both', expand=True, pady=(0, 15))
        
        # Buttons - HƏMIŞƏ AŞAĞIDA GÖRÜNƏN
        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(side='bottom', pady=15)
        
        save_btn = self.create_button(btn_frame, "💾 SAXLA", self.save_new_note, 15)
        save_btn.pack(side='left', padx=10)
        
        cancel_btn = self.create_button(btn_frame, "❌ LƏĞV ET", self.show_main_page, 15)
        cancel_btn.pack(side='left', padx=10)
    
    def save_new_note(self):
        title = self.title_entry.get().strip()
        content = self.content_text.get('1.0', tk.END).strip()
        
        if not title:
            messagebox.showwarning("Xəbərdarlıq", "Başlıq boş ola bilməz!")
            return
        
        if not content:
            messagebox.showwarning("Xəbərdarlıq", "Məzmun boş ola bilməz!")
            return
        
        self.app.save_note(title, content)
        messagebox.showinfo("Uğurlu", f"'{title}' notu saxlanıldı!")
        self.show_main_page()
    
    def view_note(self):
        selection = self.notes_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("Xəbərdarlıq", "Not seçin!")
            return
        
        if not self.notes:
            return
        
        index = selection[0]
        note = self.notes[index]
        
        self.clear_window()
        self.create_title_bar(f"📖 {note['title']}")
        
        # Info Frame
        info_frame = tk.Frame(self.root, bg=self.highlight_color, bd=2, relief='solid')
        info_frame.pack(fill='x', padx=15, pady=10)
        
        info_text = f"📅 Yaradılıb: {note['created']} | 🆔 ID: {note['id']}"
        info_label = tk.Label(
            info_frame,
            text=info_text,
            bg=self.highlight_color,
            font=('Arial', 9)
        )
        info_label.pack(pady=8, padx=10)
        
        # Content Frame
        content_frame = tk.Frame(self.root, bg='white', bd=2, relief='sunken')
        content_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        content_text = scrolledtext.ScrolledText(
            content_frame,
            font=('Arial', 11),
            bd=0,
            wrap='word',
            bg='white',
            state='normal'
        )
        content_text.pack(fill='both', expand=True, padx=5, pady=5)
        content_text.insert('1.0', note['content'])
        content_text.config(state='disabled')
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(pady=10)
        
        home_btn = self.create_button(btn_frame, "🏠 ANA SƏHİFƏ", self.show_main_page, 15)
        home_btn.pack(side='left', padx=5)
        
        delete_btn = self.create_button(
            btn_frame, 
            "🗑️ SİL", 
            lambda: self.delete_current_note(note['id']), 
            15
        )
        delete_btn.pack(side='left', padx=5)
    
    def delete_current_note(self, note_id):
        result = messagebox.askyesno("Təsdiq", "Bu notu silmək istəyirsiniz?")
        if result:
            self.app.delete_note(note_id)
            messagebox.showinfo("Uğurlu", "Not silindi!")
            self.show_main_page()
    
    def delete_note(self):
        selection = self.notes_listbox.curselection()
        
        if not selection:
            messagebox.showwarning("Xəbərdarlıq", "Silmək üçün not seçin!")
            return
        
        if not self.notes:
            return
        
        index = selection[0]
        note = self.notes[index]
        
        result = messagebox.askyesno(
            "Təsdiq", 
            f"'{note['title']}' notunu silmək istəyirsiniz?"
        )
        
        if result:
            self.app.delete_note(note['id'])
            messagebox.showinfo("Uğurlu", "Not silindi!")
            self.show_main_page()

def main():
    root = tk.Tk()
    app = RetroNotesGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()