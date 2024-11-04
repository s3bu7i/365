import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
output_path = r"C:\Users\Dino\Documents\yemek_plani.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)


# Yemək planı məlumatları DataFrame kimi
data = {
    "Gün": [
        "Bazar ertəsi", "Çərşənbə axşamı", "Çərşənbə", "Cümə axşamı", "Cümə", "Şənbə", "Bazar"
    ],
    "Səhər Yeməyi": [
        "3 yumurtalı omlet, tam taxıllı çörək, banan, qoz-fındıq",
        "Yulaf əzməsi, alma",
        "Fıstıq yağı ilə çörək, yumurta, fındıq",
        "2 yumurtalı omlet, pendir, yulaf süd ilə, alma",
        "Pendirli sandviç, qoz-fındıq, banan",
        "Yulaf, qoz-fındıq və bal, süd",
        "Pendir və yumurtalı sandviç, alma, fındıq"
    ],
    "Qəlyanaltı": [
        "Yoğurt, üzüm və ya çiyələk",
        "Kəsmik, bal və ya giləmeyvə",
        "Qatıq, üzüm və ya şaftalı",
        "Yoğurt, giləmeyvə",
        "Kəsmik, bal və giləmeyvə",
        "Yoğurt, üzüm və ya çiyələk",
        "Kəsmik, bal və giləmeyvə"
    ],
    "Nahar": [
        "Toyuq əti, qəhvəyi düyü, brokoli və kök salatı",
        "Mal əti, bulqur və ya qara düyü, qarışıq tərəvəz salatı",
        "Qızardılmış balıq, tam taxıllı makaron, göyərti salatı",
        "Toyuq, qəhvəyi düyü, tərəvəzlər",
        "Balıq, bulqur və ya qara düyü, göyərti salatı",
        "Toyuq və ya mal əti, bulqur, tərəvəz şorbası",
        "Qızardılmış balıq, qəhvəyi düyü, salat"
    ],
    "Axşamüstü Qəlyanaltı": [
        "Çörək üzərinə fıstıq yağı, süd",
        "Banan, quru meyvə",
        "Zülallı içki, çörək",
        "Çörək üzərinə avokado, süd",
        "Protein içkisi, quru meyvə",
        "Çörək üzərinə fıstıq yağı, təzə meyvə suyu",
        "Süd, qoz-fındıq"
    ],
    "Axşam Yeməyi": [
        "Qızardılmış balıq, tərəvəzli makaron, yaşıl salat",
        "Toyuq, kartof püresi, pomidor və xiyərdən salat",
        "Tərəvəzli plov, toyuq, zəngin tərəvəz salatı",
        "Mal əti, tərəvəzli makaron, salat",
        "Toyuq, kartof püresi, tərəvəz salatı",
        "Tərəvəzli plov, salat",
        "Toyuq, kartof və tərəvəzlərdən sobada yemək"
    ]
}

# DataFrame yaratmaq
df = pd.DataFrame(data)

# Qrafik hazırlamaq
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
table = plt.table(cellText=df.values, colLabels=df.columns,
                  cellLoc='center', loc='center')

# Stil və formatlama
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')

# Şəkili qeyd etmək
output_path = output_path = r"C:\Users\Dino\Documents\yemek_plani.png"

plt.savefig(output_path, bbox_inches='tight', dpi=300)
