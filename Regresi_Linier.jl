using CSV, DataFrames, CategoricalArrays, Statistics, GLM, MLJBase, MLJModels, Plots, StatsBase

# Set plotting backend
gr()  # Use GR backend for Plots.jl

# ==================== BACA DAN PREPROSES DATA LATIH ====================

data_latih = CSV.read("data_latih.csv", DataFrame)

# Tentukan kolom kategorikal
categorical_cols = [:protocol_type, :service, :flag, :label]

# Konversi ke kategori
for col in categorical_cols
    data_latih[!, col] = categorical(data_latih[!, col])
end

# Konversi label ke numerik: normal = 0, anomaly = 1
data_latih.label_num = ifelse.(data_latih.label .== "anomaly", 1.0, 0.0)

# Ambil hanya kolom numerik (tanpa label) untuk korelasi
numeric_df = DataFrames.select(data_latih, Not(categorical_cols))
cor_matrix = cor(Matrix(numeric_df))
println("Matriks korelasi:\n", cor_matrix)

# One-hot encoding menggunakan MLJModels
X_latih_raw = DataFrames.select(data_latih, Not([:label, :label_num]))

# Inisialisasi OneHotEncoder
one_hot_encoder = OneHotEncoder(features=categorical_cols[1:end-1], drop_last=false)  # Kecuali :label
mach = machine(one_hot_encoder, X_latih_raw)
fit!(mach)
X_latih = MLJBase.transform(mach, X_latih_raw)

# Gabungkan dengan label, pastikan hasilnya adalah DataFrame
df_latih = DataFrame(X_latih)
df_latih.label_num = data_latih.label_num

# Buat formula regresi linier secara dinamis
predictor_cols = names(df_latih, Not(:label_num))  # Ambil semua kolom kecuali label_num
formula_str = "label_num ~ " * join(predictor_cols, " + ")  # Buat string formula
formula = eval(Meta.parse("@formula($formula_str)"))  # Konversi ke formula

# Buat model regresi linier
model = lm(formula, df_latih)
println("\nKoefisien regresi:")
println(coef(model))

# ==================== PREDIKSI DATA LATIH ====================

y_latih_pred = GLM.predict(model, df_latih)  # Use GLM.predict
df_latih.y_pred = y_latih_pred

# Ambil nilai minimum dari prediksi yang labelnya anomaly
min_anom_pred = minimum(df_latih[df_latih.label_num .== 1.0, :y_pred])
println("\nNilai minimum prediksi yang menghasilkan label anomaly: $min_anom_pred")

# ==================== BACA DAN PREPROSES DATA UJI ====================

data_uji = CSV.read("data_uji.csv", DataFrame)

# Konversi ke kategori, sinkronkan level, dan tangani missing
for col in categorical_cols
    data_uji[!, col] = categorical(data_uji[!, col], ordered=false)  # Pastikan non-ordered
    # Konversi ke tipe yang mendukung missing
    data_uji[!, col] = convert(CategoricalVector{Union{eltype(data_uji[!, col]), Missing}}, data_uji[!, col])
    levels!(data_uji[!, col], levels(data_latih[!, col]), allowmissing=true)  # Restrict levels, allow missing
    # Ganti missing dengan level paling sering dari data latih
    if any(ismissing, data_uji[!, col])
        default_level = mode(data_latih[!, col])
        data_uji[!, col] = coalesce.(data_uji[!, col], default_level)
    end
end

# Konversi label ke numerik
data_uji.label_num = ifelse.(data_uji.label .== "anomaly", 1.0, 0.0)

# One-hot encoding untuk data uji menggunakan encoder yang sama
X_uji_raw = DataFrames.select(data_uji, Not([:label, :label_num]))
X_uji = MLJBase.transform(mach, X_uji_raw)

# Pastikan kolom X_uji sama dengan X_latih
for col in names(X_latih)
    if !(col in names(X_uji))
        X_uji[!, col] = 0.0
    end
end

# Urutkan kolom sesuai urutan di X_latih
X_uji = X_uji[:, names(X_latih)]

# Gabungkan dengan label, pastikan hasilnya adalah DataFrame
df_uji = DataFrame(X_uji)
df_uji.label_num = data_uji.label_num

# Prediksi data uji
y_uji_pred = GLM.predict(model, df_uji)  # Use GLM.predict
df_uji.y_pred = y_uji_pred

# ==================== KLASIFIKASI DAN AKURASI ====================

# Ambang batas (threshold)
threshold = min_anom_pred

# Klasifikasi
df_uji.predicted_label = ifelse.(df_uji.y_pred .>= threshold, "anomaly", "normal")

# Akurasi
actual_label = ifelse.(df_uji.label_num .== 1.0, "anomaly", "normal")
correct = sum(df_uji.predicted_label .== actual_label)
accuracy = correct / nrow(df_uji) * 100

println("\nAkurasi prediksi terhadap label asli data uji: $(round(accuracy, digits=2))%")

# ==================== VISUALISASI HASIL ====================

# 1. Confusion Matrix
cm = countmap(zip(df_uji.predicted_label, actual_label))
tn = get(cm, ("normal", "normal"), 0)
fp = get(cm, ("normal", "anomaly"), 0)
fn = get(cm, ("anomaly", "normal"), 0)
tp = get(cm, ("anomaly", "anomaly"), 0)
cm_matrix = [tn fp; fn tp]

# Plot Confusion Matrix sebagai Heatmap
p1 = heatmap(["Normal", "Anomaly"], ["Normal", "Anomaly"], cm_matrix,
             title="Confusion Matrix",
             xlabel="Predicted Label",
             ylabel="Actual Label",
             color=:blues,
             clims=(0, maximum(cm_matrix)),
             aspect_ratio=:equal,
             size=(400, 400),
             margin=5Plots.mm)

# 2. Prediction Score Distribution
normal_preds = df_uji.y_pred[df_uji.label_num .== 0.0]
anomaly_preds = df_uji.y_pred[df_uji.label_num .== 1.0]
bins = range(minimum(vcat(normal_preds, anomaly_preds)), maximum(vcat(normal_preds, anomaly_preds)), length=50)

# Plot Histogram
p2 = plot(
    histogram(normal_preds, bins=bins, label="Normal", alpha=0.6, color=:blue),
    histogram!(anomaly_preds, bins=bins, label="Anomaly", alpha=0.6, color=:red),
    title="Prediction Score Distribution",
    xlabel="Prediction Score",
    ylabel="Count",
    vline!([threshold], label="Threshold", color=:black, linestyle=:dash),
    size=(600, 400),
    margin=5Plots.mm
)

# Tampilkan plot
display(p1)
display(p2)

# Opsional: Simpan plot ke file
savefig(p1, "confusion_matrix.png")
savefig(p2, "prediction_distribution.png")