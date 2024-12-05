import cv2
import numpy as np
import matplotlib.pyplot as plt

class EloMalucoDetector:
    def __init__(self):
        pass

    def detectar_cores_quadrantes_redimensionados(self, imagem_caminho, altura_desejada=800):
        # Carregar a imagem
        imagem = cv2.imread(imagem_caminho)
        
        # Calcular a largura proporcional
        largura_desejada = int(altura_desejada * imagem.shape[1] / imagem.shape[0])

        # Tamanho final ajustado
        tamanho_final = (largura_desejada, altura_desejada)
        
        # Cortar a imagem com base no retângulo delimitador do Elo
        imagem_cortada = imagem
        
        # Redimensionar a imagem cortada para o tamanho final desejado
        imagem_redimensionada = cv2.resize(imagem_cortada, tamanho_final)

        # Converter para o espaço de cor HSV
        imagem_hsv = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2HSV)

        # Dividir a imagem em 4 quadrantes
        altura, largura, _ = imagem_redimensionada.shape
        quadrante1 = imagem_hsv[0:altura//2, 0:largura//2]  # Quadrante superior esquerdo
        quadrante2 = imagem_hsv[0:altura//2, largura//2:]  # Quadrante superior direito
        quadrante3 = imagem_hsv[altura//2:, 0:largura//2]  # Quadrante inferior esquerdo
        quadrante4 = imagem_hsv[altura//2:, largura//2:]  # Quadrante inferior direito

        # Definir intervalos de cores no espaço HSV
        # Intervalos para cada cor
        intervalo_vermelho1 = (np.array([0, 120, 70]), np.array([10, 255, 255]))
        intervalo_vermelho2 = (np.array([170, 120, 70]), np.array([180, 255, 255]))
        intervalo_verde = (np.array([35, 100, 100]), np.array([85, 255, 255]))
        intervalo_branco = (np.array([0, 0, 200]), np.array([180, 30, 255]))
        intervalo_cinza = (np.array([0, 0, 50]), np.array([180, 50, 200]))  # Cinza pode variar mais

        # Função para detectar cores dentro de um intervalo
        def detectar_cor(quadrante, intervalo_min, intervalo_max):
            mascara = cv2.inRange(quadrante, intervalo_min, intervalo_max)
            return cv2.countNonZero(mascara)

        # Detectar as cores em cada quadrante
        cores_quadrante1 = {
            "vermelho": detectar_cor(quadrante1, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante1, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante1, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante1, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante1, intervalo_cinza[0], intervalo_cinza[1])
        }

        cores_quadrante2 = {
            "vermelho": detectar_cor(quadrante2, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante2, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante2, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante2, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante2, intervalo_cinza[0], intervalo_cinza[1])
        }

        cores_quadrante3 = {
            "vermelho": detectar_cor(quadrante3, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante3, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante3, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante3, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante3, intervalo_cinza[0], intervalo_cinza[1])
        }

        cores_quadrante4 = {
            "vermelho": detectar_cor(quadrante4, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante4, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante4, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante4, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante4, intervalo_cinza[0], intervalo_cinza[1])
        }

        # Exibir os resultados para cada quadrante
        print("Cores no Quadrante 1:", cores_quadrante1)
        print("Cores no Quadrante 2:", cores_quadrante2)
        print("Cores no Quadrante 3:", cores_quadrante3)
        print("Cores no Quadrante 4:", cores_quadrante4)

        # Exibir a imagem original
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB))
        plt.title("Imagem Redimensionada")
        plt.axis("off")
        plt.show()

# Teste com uma imagem
detector = EloMalucoDetector()
detector.detectar_cores_quadrantes_redimensionados('data/Ex_input01_01.png')
