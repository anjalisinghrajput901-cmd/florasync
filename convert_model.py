from tensorflow.keras.models import load_model

# Purana model load karo
model = load_model("model/leaf_model.h5", compile=False)

# Naye Keras format me save karo
model.save("model/leaf_model.keras")

print("✅ Model converted successfully!")