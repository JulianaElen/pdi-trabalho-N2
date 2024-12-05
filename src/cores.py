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

        # Detecção de cores na imagem redimensionada
        self.detectar_cores_quadrantes(imagem_redimensionada)

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

    def detectar_cores_quadrantes(self, imagem):
        # Converter para o espaço de cor HSV
        imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

        # Dividir a imagem em 4 quadrantes
        altura, largura, _ = imagem.shape
        quadrante_superior = imagem_hsv[0:altura//4, :]  # Quadrante superior
        quadrante_medio1 = imagem_hsv[altura//4:altura//2, :]  # Quadrante médio 1
        quadrante_medio2 = imagem_hsv[altura//2:3*altura//4, :]  # Quadrante médio 2
        quadrante_inferior = imagem_hsv[3*altura//4:, :]  # Quadrante inferior

        # Definir intervalos de cores no espaço HSV
        intervalo_vermelho1 = (np.array([0, 120, 70]), np.array([10, 255, 255]))
        intervalo_vermelho2 = (np.array([170, 120, 70]), np.array([180, 255, 255]))
        intervalo_verde = (np.array([35, 100, 100]), np.array([85, 255, 255]))
        intervalo_branco = (np.array([0, 0, 200]), np.array([180, 30, 255]))
        intervalo_cinza = (np.array([0, 0, 50]), np.array([180, 50, 200]))
        intervalo_amarelo = (np.array([30, 100, 100]), np.array([50, 255, 255]))  # Amarelo

        # Função para detectar cores dentro de um intervalo
        def detectar_cor(quadrante, intervalo_min, intervalo_max):
            mascara = cv2.inRange(quadrante, intervalo_min, intervalo_max)
            return cv2.countNonZero(mascara)

        # Detectar as cores em cada quadrante
        cores_quadrante_superior = {
            "vermelho": detectar_cor(quadrante_superior, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante_superior, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante_superior, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante_superior, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante_superior, intervalo_cinza[0], intervalo_cinza[1]),
            "amarelo": detectar_cor(quadrante_superior, intervalo_amarelo[0], intervalo_amarelo[1])
        }

        cores_quadrante_medio1 = {
            "vermelho": detectar_cor(quadrante_medio1, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante_medio1, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante_medio1, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante_medio1, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante_medio1, intervalo_cinza[0], intervalo_cinza[1]),
            "amarelo": detectar_cor(quadrante_medio1, intervalo_amarelo[0], intervalo_amarelo[1])
        }

        cores_quadrante_medio2 = {
            "vermelho": detectar_cor(quadrante_medio2, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante_medio2, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante_medio2, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante_medio2, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante_medio2, intervalo_cinza[0], intervalo_cinza[1]),
            "amarelo": detectar_cor(quadrante_medio2, intervalo_amarelo[0], intervalo_amarelo[1])
        }

        cores_quadrante_inferior = {
            "vermelho": detectar_cor(quadrante_inferior, intervalo_vermelho1[0], intervalo_vermelho1[1]) + detectar_cor(quadrante_inferior, intervalo_vermelho2[0], intervalo_vermelho2[1]),
            "verde": detectar_cor(quadrante_inferior, intervalo_verde[0], intervalo_verde[1]),
            "branco": detectar_cor(quadrante_inferior, intervalo_branco[0], intervalo_branco[1]),
            "cinza": detectar_cor(quadrante_inferior, intervalo_cinza[0], intervalo_cinza[1]),
            "amarelo": detectar_cor(quadrante_inferior, intervalo_amarelo[0], intervalo_amarelo[1])
        }

        # Exibir os resultados para cada quadrante
        print("Cores no Quadrante Superior:", cores_quadrante_superior)
        print("Cores no Quadrante Médio 1:", cores_quadrante_medio1)
        print("Cores no Quadrante Médio 2:", cores_quadrante_medio2)
        print("Cores no Quadrante Inferior:", cores_quadrante_inferior)

# Teste com uma imagem e um ponto destino fixo
detector = EloMalucoDetector()

# Ponto fixo para centralizar o Elo (exemplo: o centro da imagem)
ponto_destino = (200, 250)

detector.detectar_bordas('data/Ex_input01_01.png', ponto_destino)
detector.detectar_bordas('data/Ex_input01_02.png', ponto_destino)
detector.detectar_bordas('data/Ex_input01_03.png', ponto_destino)
detector.detectar_bordas('data/Ex_input01_04.png', ponto_destino)

