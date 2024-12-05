import cv2
import numpy as np
import matplotlib.pyplot as plt

class EloMalucoDetector:
    def __init__(self):
        pass

    def detectar_bordas(self, imagem_caminho, ponto_destino, altura_desejada=800):
        # Carregar a imagem
        imagem = cv2.imread(imagem_caminho)
        
        # Calcular a largura proporcional
        largura_desejada = int(altura_desejada * imagem.shape[1] / imagem.shape[0])

        # Tamanho final ajustado
        tamanho_final = (largura_desejada, altura_desejada)
        
        # Converter para escala de cinza
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Suavizar para reduzir ruídos
        imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
        
        # Detectar bordas usando o detector de Canny
        bordas = cv2.Canny(imagem_suavizada, 50, 150)

        # Detectar contornos (para localizar o Elo)
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            print("Nenhum contorno encontrado!")
            return bordas

        # Encontrar o maior contorno (assumindo que é o Elo)
        maior_contorno = max(contornos, key=cv2.contourArea)
        
        # Encontrar o retângulo delimitador do Elo
        x, y, w, h = cv2.boundingRect(maior_contorno)

        # Calcular o centro do Elo na imagem
        centro_elo = (x + w // 2, y + h // 2)

        # Calcular o deslocamento necessário para centralizar o Elo no ponto destino
        deslocamento_x = ponto_destino[0] - centro_elo[0]
        deslocamento_y = ponto_destino[1] - centro_elo[1]

        # Criar a matriz de transformação para translação
        matriz_translacao = np.float32([[1, 0, deslocamento_x], [0, 1, deslocamento_y]])

        # Aplicar a translação nas bordas detectadas
        bordas_transladadas = cv2.warpAffine(bordas, matriz_translacao, (imagem.shape[1], imagem.shape[0]))

        # Cortar a imagem com base no retângulo delimitador
        imagem_cortada = imagem[y:y+h, x:x+w]
        
        # Redimensionar a imagem cortada para o tamanho final desejado
        imagem_redimensionada = cv2.resize(imagem_cortada, tamanho_final)

        # Redimensionar as bordas transladadas para o mesmo tamanho
        bordas_redimensionadas = cv2.resize(bordas_transladadas, tamanho_final)

        # Exibir as imagens na ordem solicitada
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Bordas Detectadas")
        plt.imshow(bordas, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Bordas Transladadas")
        plt.imshow(bordas_redimensionadas, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Imagem Cortada e Redimensionada")
        plt.imshow(cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.show()

        return bordas_redimensionadas

# Teste com uma imagem e um ponto destino fixo
detector = EloMalucoDetector()

# Ponto fixo para centralizar o Elo (exemplo: o centro da imagem)
ponto_destino = (200, 250)

detector.detectar_bordas('data/Ex_input01_01.png', ponto_destino)
detector.detectar_bordas('data/Ex_input01_02.png', ponto_destino)
detector.detectar_bordas('data/Ex_input01_03.png', ponto_destino)
detector.detectar_bordas('data/Ex_input01_04.png', ponto_destino)
