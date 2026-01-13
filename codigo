import face_recognition
import numpy as np

# 1) Carrega a face conhecida
known_img = face_recognition.load_image_file("known/user.jpg")
known_encodings = face_recognition.face_encodings(known_img)

if len(known_encodings) == 0:
    raise ValueError("Nenhum rosto detectado em known/user.jpg")

known_embedding = known_encodings[0]

# 2) Carrega a imagem para validar
test_img = face_recognition.load_image_file("test.jpg")
test_encodings = face_recognition.face_encodings(test_img)

if len(test_encodings) == 0:
    print("Sem rosto na imagem de teste.")
    raise SystemExit

test_embedding = test_encodings[0]

# 3) Distância + limiar
dist = np.linalg.norm(known_embedding - test_embedding)

# Limiar típico: 0.6 (pode ajustar)
THRESHOLD = 0.6
is_same = dist < THRESHOLD

print(f"Distância: {dist:.4f}")
print("MESMA PESSOA ✅" if is_same else "PESSOA DIFERENTE ❌")
