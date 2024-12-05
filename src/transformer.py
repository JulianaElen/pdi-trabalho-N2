# Classe para normalizar a imagem
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageTransformer:
    def __init__(self, tamanho=(300, 300)):
        self.tamanho = tamanho  # Define o tamanho para o redimensionamento da imagem
    
    def redimensionar(self, imagem):
        # Redimensionar a imagem para o tamanho padrão
        imagem_redimensionada = cv2.resize(imagem, self.tamanho)
        return imagem_redimensionada

    def normalizar(self, imagem):
        # Normalizar a imagem, convertendo os valores de pixel para o intervalo [0, 1]
        imagem_normalizada = imagem / 255.0  # Divide pelos valores máximos de pixel (255) para normalizar
        return imagem_normalizada

    def redimensionar_e_normalizar(self, imagem):
        # Combina o redimensionamento e a normalização em um único passo
        imagem_redimensionada = self.redimensionar(imagem)
        imagem_normalizada = self.normalizar(imagem_redimensionada)
        
        # Exibição das imagens (original e redimensionada)
        self.visualizar(imagem, imagem_redimensionada)
        
        return imagem_normalizada

    def visualizar(self, imagem_original, imagem_redimensionada):
        # Verificar se a imagem está sendo carregada corretamente
        if imagem_original is None or imagem_redimensionada is None:
            print("Erro: uma das imagens não foi carregada corretamente.")
            return
        
        # Mostrar as imagens lado a lado para comparação
        plt.figure(figsize=(12, 6))

        # Exibir imagem original
        plt.subplot(1, 2, 1)
        plt.title("Imagem Original")
        plt.imshow(cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        # Exibir imagem redimensionada
        plt.subplot(1, 2, 2)
        plt.title("Imagem Redimensionada")
        plt.imshow(cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        # Mostrar as imagens
        plt.show()