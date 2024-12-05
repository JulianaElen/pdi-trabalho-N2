import cv2
import matplotlib.pyplot as plt


# Detetar as bordas do elo
class EloMalucoDetector:
    def __init__(self):
        pass

    def detectar_bordas(self, imagem_caminho): 
        # Carregar a imagem
        imagem = cv2.imread(imagem_caminho)
        
        # Converter para escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Suavizar para reduzir ruídos
        imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
        
        # Detectar bordas usando o detector de Canny
        bordas = cv2.Canny(imagem_suavizada, 50, 150)
        
        # Exibir as imagens em etapas
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Imagem Original")
        plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.title("Imagem em Escala de Cinza")
        plt.imshow(imagem_cinza, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.title("Bordas Detectadas")
        plt.imshow(bordas, cmap="gray")
        plt.axis("off")
        
        plt.show()
    
detector = EloMalucoDetector()
# teste usando detector.detectar_bordas('data/Ex_input01_01.png')
