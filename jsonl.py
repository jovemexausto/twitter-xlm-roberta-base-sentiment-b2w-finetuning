import pandas as pd
import json
import os

def convert_jsonl_to_sentiment_csv(jsonl_path, output_csv_path):
    """
    Converte um arquivo JSONL de texto e sentimento para um arquivo CSV
    com as colunas 'polarity' e 'text'.

    Mapeamento de sentimento para polaridade:
    - 'neutral': 1
    - 'negative': 0
    - 'positive': 2

    Args:
        jsonl_path (str): Caminho para o arquivo de entrada .jsonl.
        output_csv_path (str): Caminho para o arquivo de saída .csv.
    """
    # Mapeamento de sentimentos para valores de polaridade numérica
    sentiment_to_polarity = {
        "neutral": 1,
        "negative": 0,
        "positive": 2
    }

    texts = []
    polarities = []
    processed_count = 0

    # Verifica se o arquivo JSONL de entrada existe
    if not os.path.exists(jsonl_path):
        print(f"Erro: Arquivo JSONL de entrada não encontrado em '{jsonl_path}'.")
        return

    print(f"Iniciando a conversão de '{jsonl_path}' para CSV...")

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = data.get("text")
                    sentiment = data.get("sentiment")

                    # Garante que 'text' e 'sentiment' existem e não são vazios
                    if text is None or sentiment is None:
                        print(f"Aviso: Linha {line_num} no JSONL está faltando 'text' ou 'sentiment'. Pulando.")
                        continue

                    # Converte o sentimento para polaridade usando o mapeamento
                    polarity = sentiment_to_polarity.get(sentiment.lower())

                    if polarity is None:
                        print(f"Aviso: Sentimento desconhecido '{sentiment}' na linha {line_num}. Pulando.")
                        continue

                    texts.append(text)
                    polarities.append(polarity)
                    processed_count += 1

                except json.JSONDecodeError as e:
                    print(f"Erro de decodificação JSON na linha {line_num}: {e}. Linha: '{line.strip()}'")
                except Exception as e:
                    print(f"Erro inesperado ao processar a linha {line_num}: {e}. Linha: '{line.strip()}'")

    except Exception as e:
        print(f"Erro ao ler o arquivo JSONL '{jsonl_path}': {e}")
        return

    if not texts:
        print("Nenhum dado válido foi processado para o CSV. Verifique o arquivo JSONL.")
        return

    # Cria um DataFrame Pandas
    df_output = pd.DataFrame({
        'polarity': polarities,
        'text': texts
    })

    # Garante que o diretório de saída exista
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva o DataFrame em um arquivo CSV
    try:
        df_output.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Conversão concluída! {processed_count} registros salvos em '{output_csv_path}'.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo CSV '{output_csv_path}': {e}")

if __name__ == "__main__":
    # Define o caminho do arquivo JSONL de entrada
    input_jsonl_file = './data/neutral.jsonl' # Altere para o caminho do seu arquivo JSONL

    # Define o caminho do arquivo CSV de saída
    output_csv_file = './data/output_sentiment_data.csv' # Você pode ajustar este caminho

    # Chama a função principal para converter o JSONL para CSV
    convert_jsonl_to_sentiment_csv(input_jsonl_file, output_csv_file)
