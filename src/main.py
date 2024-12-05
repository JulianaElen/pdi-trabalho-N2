from detector import EloMalucoDetector
from transformer import ImageTransformer
#from analyzer import EloAnalyzer
#from templates import loadTemplates
#from xml_writer import XMLWriter
import cv2
import matplotlib.pyplot as plt
def main():
    # Configurações iniciais
    input_images = [
        "data/Ex_input01_01.png",
        "data/Ex_input01_02.png",
        "data/Ex_input01_03.png",
        "data/Ex_input01_04.png",
    ]
    output_xml = "estado_atual.xml"
    ponto_destino = (400, 300)
    # Inicializar as classes principais
    detector = EloMalucoDetector()
    transformer = ImageTransformer()
    #analyzer = EloAnalyzer()
    #templates = loadTemplates()
    #xml_writer = XMLWriter()

    # Lista para armazenar os estados de cada face
    estado_jogo = []

    # Processar cada imagem
    for img_path in input_images:
        print(f"Processando {img_path}...")

        # 1. Carregar e detectar bordas
        bordas = detector.detectar_bordas(img_path, ponto_destino)

        # 2. Redimensionar e normalizar a imagem
        imagem_normalizada = transformer.redimensionar(bordas)

        # Exibir as imagens para verificação
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Imagem Original: {img_path}")
        plt.imshow(cv2.cvtColor(bordas, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.title("Imagem Redimensionada")
        plt.imshow(cv2.cvtColor(imagem_normalizada, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        
        plt.show()

        # 3. Analisar cores e silhuetas
        #cores, silhuetas = analyzer.analisar(imagem_normalizada, templates)

        # 4. Adicionar o estado da face à lista
        #estado_jogo.append((cores, silhuetas))

    # Gerar o arquivo XML com o estado do jogo
    #xml_writer.gerar_xml(estado_jogo, output_xml)
    print(f"Arquivo XML gerado: {output_xml}")

if __name__ == "__main__":
    main()
