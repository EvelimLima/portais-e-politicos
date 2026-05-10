-- =============================================================
-- schema.sql — Portais e Políticos
-- Banco de dados: sbbd_dados
-- Executar: mysql -u root -p < schema.sql
-- =============================================================

CREATE DATABASE IF NOT EXISTS sbbd_dados
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE sbbd_dados;

-- -------------------------------------------------------------
-- Tabela 1: Configuração das Fontes RSS
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS config_feeds (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    nome_site     VARCHAR(100) NOT NULL,
    url_rss       VARCHAR(255) NOT NULL UNIQUE,
    ativo         BOOLEAN DEFAULT TRUE,
    data_cadastro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ------------------------------------------------------------- 
-- Tabela 2: Políticos Monitorados
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS config_politicos (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    nome          VARCHAR(150) NOT NULL UNIQUE,
    aliases       TEXT         NOT NULL,
    ativo         BOOLEAN      DEFAULT TRUE,
    data_cadastro TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- -------------------------------------------------------------
-- Tabela 3: Notícias Coletadas e Limpas
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS noticias_processadas (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    portal          VARCHAR(100),
    link            VARCHAR(255) UNIQUE,
    titulo          TEXT,
    data_publicacao DATETIME,
    politico_busca  VARCHAR(100),
    conteudo_limpo  LONGTEXT,
    data_coleta     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- -------------------------------------------------------------
-- Tabela 4: Análises de Sentimento
-- -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS analises_llama (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    noticia_id   INT,
    politico     VARCHAR(100),
    trecho       TEXT,
    classificacao VARCHAR(20),
    motivo       TEXT,
    modelo       VARCHAR(100),
    data_analise TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (noticia_id) REFERENCES noticias_processadas(id) ON DELETE CASCADE,
    CONSTRAINT uq_noticia_modelo UNIQUE (noticia_id, modelo)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;