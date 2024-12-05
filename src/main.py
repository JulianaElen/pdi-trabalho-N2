from detector import EloMalucoDetector
#from transformer import ImageTransformer
#from analyzer import EloAnalyzer
#from templates import loadTemplates
#from xml_writer import XMLWriter

def main():
    # Configurações iniciais
    input_images = [
        "data/Ex_input01_01.png",
        "data/Ex_input01_02.png",
        "data/Ex_input01_03.png",
        "data/Ex_input01_04.png",
    ]
    output_xml = "estado_atual.xml"

    # Inicializar as classes principais
    detector = EloMalucoDetector()
    #transformer = ImageTransformer()
    #analyzer = EloAnalyzer()
    #templates = loadTemplates()
    #xml_writer = XMLWriter()

    # Lista para armazenar os estados de cada face
    estado_jogo = []

    # Processar cada imagem
    for img_path in input_images:
        print(f"Processando {img_path}...")

        # 1. Carregar e detectar bordas
        bordas = detector.detectar_bordas(img_path)

        # 2. Redimensionar e normalizar a imagem
        #imagem_normalizada = transformer.redimensionar(bordas)

        # 3. Analisar cores e silhuetas
        #cores, silhuetas = analyzer.analisar(imagem_normalizada, templates)

        # 4. Adicionar o estado da face à lista
        #estado_jogo.append((cores, silhuetas))

    # Gerar o arquivo XML com o estado do jogo
    #xml_writer.gerar_xml(estado_jogo, output_xml)
    print(f"Arquivo XML gerado: {output_xml}")

if __name__ == "__main__":
    main()
