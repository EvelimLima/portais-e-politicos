# coletor_sbbd.py
import config
import os
import time
import random
import logging
import feedparser
import pandas as pd
import requests
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Comment
import re
from sqlalchemy import create_engine, text
import sys

# ==========================================
# CORREÇÃO #1: logging em vez de redirecionar sys.stdout globalmente
# O Logger antigo substituía sys.stdout no nível de módulo, quebrando
# o Streamlit sempre que fazia "import coletor_sbbd".
# ==========================================
pasta_do_script = os.path.dirname(os.path.realpath(__file__))
log_file_path = os.path.join(pasta_do_script, 'coleta_log.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# CORREÇÃO #11 (parte coletor): função utilitária de parse centralizada
# ==========================================
def parsear_resposta_llama(resposta: str) -> tuple[str, str]:
    """
    Extrai (motivo, classificacao) da resposta do modelo.
    Centralizada aqui para ser importada tanto pelo coletor quanto pelo app.
    """
    motivo_match = re.search(
        r'Motivo:\s*(.*?)(?:Classifica[cç][aã]o:|$|\n)',
        resposta, re.IGNORECASE | re.DOTALL
    )
    classif_match = re.search(
        r'Classifica[cç][aã]o:\s*([A-Za-z]+)',
        resposta, re.IGNORECASE
    )

    motivo = motivo_match.group(1).strip() if motivo_match else "Não foi possível extrair o motivo"
    classificacao = classif_match.group(1).strip().capitalize() if classif_match else "Neutro"

    # Normaliza variações em inglês ou truncadas
    mapa = {"Positive": "Positivo", "Negative": "Negativo", "Neutral": "Neutro"}
    classificacao = mapa.get(classificacao, classificacao)

    if classificacao not in ("Positivo", "Negativo", "Neutro"):
        cl = classificacao.lower()
        if "positiv" in cl:
            classificacao = "Positivo"
        elif "negativ" in cl:
            classificacao = "Negativo"
        else:
            classificacao = "Neutro"

    return motivo, classificacao


# ==========================================
# CORREÇÃO #5: helper de requisição com retry e backoff exponencial
# ==========================================
def _fazer_requisicao_com_retry(url: str, headers: dict, max_tentativas: int = 3):
    """
    Tenta buscar a URL com cloudscraper (se disponível) ou requests puro.
    Em caso de erro 429/503, aplica backoff exponencial antes de retentar.
    Cada tentativa esgota cloudscraper E requests antes de avançar.
    """
    ultimo_erro = None
    for tentativa in range(1, max_tentativas + 1):
        delay = random.uniform(1, 3) * (2 ** (tentativa - 1))  # 1-3s, 2-6s, 4-12s
        time.sleep(delay)

        try:
            import cloudscraper
            scraper = cloudscraper.create_scraper(
                browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
            )
            resp = scraper.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            logger.info(f"[cloudscraper] OK (tentativa {tentativa}): {url[:70]}")
            return resp
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"[cloudscraper] tentativa {tentativa} falhou: {e}")
            ultimo_erro = e

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            logger.info(f"[requests] OK (tentativa {tentativa}): {url[:70]}")
            return resp
        except Exception as e:
            logger.warning(f"[requests] tentativa {tentativa} falhou: {e}")
            ultimo_erro = e

    raise RuntimeError(f"Todas as {max_tentativas} tentativas falharam para {url}") from ultimo_erro


# --- CLASSE DO COLETOR ---
class ColetorParaMySQL:

    def __init__(self, politicos, db_user, db_pass, db_name="sbbd_dados", db_host='localhost'):
        self.politicos = politicos

        try:
            connection_string = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}/{db_name}"
            self.engine = create_engine(connection_string)
            logger.info(f"Conexão com o banco '{db_name}' estabelecida com sucesso.")
        except Exception as e:
            logger.error(f"ERRO CRÍTICO: Não foi possível conectar ao MySQL. Erro: {e}")
            self.engine = None

        self.HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.google.com/'
        }

        self.TAGS_RUIDO_GLOBAIS = [
            'script', 'style', 'nav', 'footer', 'header',
            'iframe', 'noscript', 'aside', 'form', 'figure'
        ]

        self.CLASSES_RUIDO_GLOBAIS = [
            'sharedaddy', 'share-buttons', 'social-share', 'addtoany',
            'comments-area', 'comment-respond', 'wp-comment',
            'post-navigation', 'nav-links', 'prev-next',
            'related-posts', 'jp-relatedposts', 'yarpp',
            'newsletter', 'popup', 'subscribe',
            'advertisement', 'ad-wrap', 'ads-container', 'google-auto-placed',
            'breadcrumb', 'breadcrumbs',
        ]

        self.SELETORES_POR_DOMINIO = {
            "amazonasatual.com.br": {
                "conteudo": ("div", "td-post-content"),
                "ruido_classes": [
                    "td_block_related_posts", "td-block-title-wrap", "td-block-title",
                    "td_block_trending_now", "td-ss-main-sidebar", "td-post-sharing",
                    "td-post-sharing-top", "td-post-sharing-bottom", "td-comments-area",
                    "td-post-next-prev", "td-a-rec", "td-g-rec",
                ]
            },
            "bncamazonas.com.br": {
                "conteudo": ("div", "content-single"),
                "ruido_classes": []
            },
            "emtempo.com.br": {
                "conteudo": ("div", "td-post-content"),
                "ruido_classes": [
                    "td_block_related_posts", "td-block-title-wrap",
                    "td-post-sharing", "td-comments-area",
                ]
            },
            "deamazonia.com.br": {
                "conteudo": ("div", "entry-content"),
                "ruido_classes": [
                    "post-tags", "post-categories", "sharedaddy",
                    "related-posts", "nav-links", "post-navigation",
                ]
            },
        }

        self.USER_AGENTS = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
        ]

    def buscar_feeds_ativos_do_banco(self):
        if not self.engine:
            return {}
        try:
            query = "SELECT nome_site, url_rss FROM config_feeds WHERE ativo = 1"
            df = pd.read_sql(query, self.engine)
            feeds_ativos = dict(zip(df.url_rss, df.nome_site))
            logger.info(f"Encontrados {len(feeds_ativos)} portais ativos no banco de dados.")
            return feeds_ativos
        except Exception as e:
            logger.error(f"Erro ao buscar feeds: {e}")
            return {}

    def _consertar_acentos(self, texto):
        if not isinstance(texto, str):
            return texto
        try:
            texto = texto.encode('latin1').decode('utf-8')
        except Exception:
            pass

        correcoes = {
            "Ã¡": "á", "Ã ": "á", "Ã\xa0": "á", "Ã gua": "água", "terÃ ": "terá",
            "Ã©": "é", "Ã\xa9": "é", "tambÃ m": "também", "SÃ rgio": "Sérgio",
            "Ãª": "ê", "Ã\xaa": "ê", "Ã³": "ó", "Ã\xb3": "ó", "Ãº": "ú",
            "Ã§": "ç", "Ã\xa7": "ç", "Ã£": "ã", "Ã\xa3": "ã",
            "Ã§Ã£o": "ção", "populaÃ Ã o": "população", "Ã-": "í", "Ã\xad": "í"
        }
        for errado, certo in correcoes.items():
            texto = texto.replace(errado, certo)
        return texto

    def _pipeline_de_limpeza(self, texto_bruto):
        if not isinstance(texto_bruto, str):
            return ""
        texto_limpo = self._consertar_acentos(texto_bruto)
        texto_limpo = re.sub(r'http\S+|www\S+', '', texto_limpo)

        padroes_lixo = [
            # ── Portais mapeados ──────────────────────────────────────────────
            r"Ao navegar por nosso conteúdo.*?Confirmo \d{4}",
            r"Amazonas Atual Dia a Dia.*?Compartilhar",
            r"Aa Inicial.*?Quem Somos( Pesquisar Inicial.*?Quem Somos Siga-nos.*?Quem Somos)?",
            r"Fechar menu lateral.*?Buscar conteúdo Youtube Facebook Twitter Instagram",
            r"Últimas Notícias Amazonas.*?Início Política",
            r"\(\s*Do ATUAL MANAUS\s*-?\s*\)?",
            r"\(?\s*\)?\s*Com informações da assessoria",
            r"Compartilhar Publicado em: \d{2}/\d{2}/\d{4}.*?Atualizado em: \d{2}/\d{2}/\d{4}.*?\d{2}:\d{2}",
            r"Compartilhe:\s*Tags:.*?(?=Posts Relacionados|$)",
            r"Notícias relacionadas.*?(?=Compartilhe|Deixe um comentário|$)",
            r"Compartilhe Facebook Twitter.*?Nome E-mail Site( ×.*?20)?",
            r"Facebook WhatsApp Telegram Twitter Manaus Alerta",
            r"O Portal EM TEMPO é a plataforma de notícias digitais.*?Todos os direitos reservados",
            r"\d+\s*[−\-]\s*=\s*\d+",
            r"Dia a Dia\s+[A-Z][^.]{10,80}\d{1,2} de \w+ de \d{4}",
            # ── Genéricos — qualquer portal ──────────────────────────────────
            r"Veja também[:\s]*",
            r"Leia também[:\s]*",
            r"Artigos?\s+relacionados.*?(?=\n\n|\Z)",
            r"Posts?\s+relacionados.*?(?=\n\n|\Z)",
            r"Siga-?nos\b.*?(?=\n|$)",
            r"Compartilhe\s+audio\s+by\s+\w+",
            r"audio\s+by\s+\w+",
            r"Podcast\s+\w+",
            r"Grupo\s+(Telegram|Whatsapp|WhatsApp|Facebook)",
            r"Por\s+\w+\s+em\s+\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\s+às\s+\d{2}:\d{2}",
            r"\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\s+às\s+\d{2}:\d{2}",
            r"Publicado\s+em:?\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
            r"Atualizado\s+em:?\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
            r"\[\s*\.\.\.\s*\]",
            r"Tags?:\s*[\w\s,#áàãâéêíóôõúç]+(?=\n|$)",
            r"Categorias?:\s*[\w\s,áàãâéêíóôõúç]+(?=\n|$)",
            r"Fonte:\s*.{0,80}(?=\n|$)",
            r"Da\s+Redação[\s\-]*",
            r"\(\s*Da\s+Redação\s*\)",
            r"Todos os direitos reservados.*?(?=\n|$)",
            r"© ?\d{4}.*?(?=\n|$)",
            r"Compartilhe\s*[←→].*?(?=\n|\Z)",
            r"[←→]\s*[A-ZÁÀÃÂÉÊÍÓÔÕÚ][^\n]{10,120}",
            r"(Destaque|Bastidores|Política|Economia|Esporte|Saúde|Cultura)"
            r"\s+\w+\s+\d{1,2},?\s+\d{4}\s+\w+\s+\d{1,2},?\s+\d{4}\s+\w+",
            r"\bdeamazonia\b",
        ]

        for p in padroes_lixo:
            texto_limpo = re.sub(p, " ", texto_limpo, flags=re.IGNORECASE | re.DOTALL)

        texto_limpo = re.sub(r'\.{3,}', '. ', texto_limpo)
        texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
        return texto_limpo

    def _extrair_texto_visivel(self, soup):
        def visivel(elemento):
            if elemento.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
                return False
            if isinstance(elemento, Comment):
                return False
            return True
        textos = soup.find_all(string=True)
        return " ".join(t.strip() for t in filter(visivel, textos) if t.strip())

    def _remover_ruido_html(self, soup, classes_extras=None):
        for tag in self.TAGS_RUIDO_GLOBAIS:
            for el in soup.find_all(tag):
                el.decompose()
        todas_classes_ruido = self.CLASSES_RUIDO_GLOBAIS + (classes_extras or [])
        for classe in todas_classes_ruido:
            for el in soup.find_all(class_=lambda c: c and classe in c):
                el.decompose()

    def _fallback_generico(self, soup):
        candidatos_seletores = [
            soup.find("article"),
            soup.find("main"),
            soup.find("div", {"id": re.compile(r"content|article|post|body|texto|noticia", re.I)}),
            soup.find("div", class_=re.compile(r"content|article|post|body|texto|noticia|entry", re.I)),
        ]
        candidatos_validos = [c for c in candidatos_seletores if c is not None]

        if candidatos_validos:
            melhor = max(candidatos_validos, key=lambda el: len(el.get_text(strip=True)))
            texto = self._extrair_texto_visivel(melhor)
            if len(texto) > 200:
                return texto

        paragrafos = soup.find_all("p")
        texto_paragrafos = " ".join(
            p.get_text(strip=True) for p in paragrafos
            if len(p.get_text(strip=True)) > 60
        )
        if len(texto_paragrafos) > 200:
            return texto_paragrafos

        return self._extrair_texto_visivel(soup)

    def _fazer_requisicao(self, url):
        headers = {**self.HEADERS, 'User-Agent': random.choice(self.USER_AGENTS)}
        # CORREÇÃO #5: delega ao helper com retry e backoff
        return _fazer_requisicao_com_retry(url, headers)

    def obter_conteudo_com_requests(self, url):
        try:
            response = self._fazer_requisicao(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            dominio = urlparse(url).netloc

            config_dominio = None
            for dominio_chave, cfg in self.SELETORES_POR_DOMINIO.items():
                if dominio_chave in dominio:
                    config_dominio = cfg
                    break

            classes_ruido_extras = config_dominio.get("ruido_classes", []) if config_dominio else []
            self._remover_ruido_html(soup, classes_ruido_extras)

            seletor_principal = None
            if config_dominio:
                tag, classe = config_dominio["conteudo"]
                seletor_principal = soup.find(tag, class_=classe)

            if seletor_principal:
                texto = self._extrair_texto_visivel(seletor_principal)
                if len(texto) > 200:
                    return texto

            logger.info(f"Usando fallback genérico para: {dominio}")
            return self._fallback_generico(soup)

        except Exception as e:
            logger.error(f"Erro ao processar {url}: {e}")
            return None

    def extrair_noticias(self):
        """
        CORREÇÃO #2: gera uma linha por político encontrado no artigo,
        não apenas o primeiro.
        CORREÇÃO #3: o pré-filtro RSS coleta todos os políticos mencionados
        no título/resumo, sem break precoce.
        CORREÇÃO #4: links_processados é consultado incrementalmente via banco
        em vez de um set ilimitado em memória.
        CORREÇÃO #6: artigos com conteúdo limpo muito curto são descartados.
        """
        if not self.engine:
            return []

        feeds_ativos = self.buscar_feeds_ativos_do_banco()
        if not feeds_ativos:
            return []

        # CORREÇÃO #4: busca links já existentes do banco (sem manter set crescente)
        try:
            df_antigo = pd.read_sql("SELECT link FROM noticias_processadas", self.engine)
            links_processados = set(df_antigo['link'].tolist())
        except Exception:
            links_processados = set()

        MIN_CONTEUDO = 150  # CORREÇÃO #6: mínimo de caracteres para aceitar o artigo

        todas_noticias = []
        for url_feed, nome_portal in feeds_ativos.items():
            logger.info(f"Verificando feed: {nome_portal}")
            feed = feedparser.parse(url_feed)

            for entry in feed.entries:
                link = entry.link
                if link in links_processados:
                    continue

                # CORREÇÃO #3: pré-filtro sem break — coleta todos os políticos
                # mencionados no título/resumo
                texto_rapido = f"{entry.title} {entry.get('summary', '')}".lower()
                politicos_pre = [
                    nome for nome, aliases in self.politicos.items()
                    if any(alias.lower() in texto_rapido for alias in aliases)
                ]

                if not politicos_pre:
                    links_processados.add(link)
                    continue

                logger.info(f"Extraindo: {entry.title[:60]}...")
                conteudo_bruto = self.obter_conteudo_com_requests(link)
                links_processados.add(link)

                if not conteudo_bruto:
                    continue

                conteudo_limpo = self._pipeline_de_limpeza(conteudo_bruto)

                # CORREÇÃO #6: descarta artigos com conteúdo muito curto
                if len(conteudo_limpo) < MIN_CONTEUDO:
                    logger.warning(f"Conteúdo insuficiente ({len(conteudo_limpo)} chars), descartando: {link}")
                    continue

                # CORREÇÃO #2: verifica todos os políticos no corpo do artigo,
                # não apenas o primeiro encontrado
                data_pub = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    data_pub = datetime(*entry.published_parsed[:6])

                politicos_encontrados = [
                    nome for nome, aliases in self.politicos.items()
                    if any(alias.lower() in conteudo_limpo.lower() for alias in aliases)
                ]

                if not politicos_encontrados:
                    continue

                for nome_politico in politicos_encontrados:
                    logger.info(f"ENCONTRADO: Menção a '{nome_politico}'.")
                    todas_noticias.append({
                        'portal': nome_portal,
                        'link': link,
                        'titulo': entry.title,
                        'data_publicacao': data_pub,
                        'politico_busca': nome_politico,
                        'conteudo_limpo': conteudo_limpo
                    })

        return todas_noticias

    def executar_coleta_e_processamento_incremental(self):
        if not self.engine:
            return
        logger.info(f"=== Iniciando Coleta Incremental: {datetime.now().strftime('%d/%m/%Y %H:%M')} ===")

        noticias_novas = self.extrair_noticias()

        if not noticias_novas:
            logger.info("Nenhuma notícia nova encontrada. Processo finalizado.")
            return

        logger.info(f"Salvando {len(noticias_novas)} notícias na tabela 'noticias_processadas'...")
        df_limpas = pd.DataFrame(noticias_novas)
        df_limpas['data_publicacao'] = pd.to_datetime(df_limpas['data_publicacao'], errors='coerce')
        df_limpas.to_sql('noticias_processadas', con=self.engine, if_exists='append', index=False)
        logger.info("=== Processo finalizado com sucesso! ===")


if __name__ == "__main__":
    engine_main = create_engine(f"mysql+pymysql://{config.USUARIO_MYSQL}:{config.SENHA_MYSQL}@{config.HOST_MYSQL}:{config.PORTA_MYSQL}/{config.BANCO_MYSQL}")

    df_pol = pd.read_sql("SELECT nome, aliases FROM config_politicos WHERE ativo = 1", engine_main)
    politicos_do_banco = {}
    for _, row in df_pol.iterrows():
        aliases = [a.strip() for a in str(row['aliases']).split(',') if a.strip()]
        politicos_do_banco[row['nome']] = aliases

    coletor = ColetorParaMySQL(
        politicos=politicos_do_banco,
        db_user=config.USUARIO_MYSQL,
        db_pass=config.SENHA_MYSQL,
        db_name=config.BANCO_MYSQL,
        db_host=config.HOST_MYSQL
    )
    coletor.executar_coleta_e_processamento_incremental()