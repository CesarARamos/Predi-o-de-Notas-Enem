# Predição de Notas

  O contexto do desafio gira em torno dos resultados do ENEM 2016 (disponíveis no arquivo train.csv) fornecidos pela Covenation junto com um dicionário de dados em EXCEL e outro arquivo CSV utilizado para predição dos valores. 
O projeto contava com uma base de dados de 167 variáveis e mais de 13 mil observações cujo objetivo final era a previsão da nota de matemática de um aluno e obrigatoriamente o algoritmo deveria ter uma eficácia maior que 90%.
Com isso, vamos ao que interessa:
```r
# Definindo a pasta de trabalho
setwd("~/codenation")
```
```r
# Carregando os pacotes 
library(Amelia)
library(ggplot2)
library(caret)
library(reshape)
library(randomForest)
library(dplyr)
library(e1071)
library(GGally)
library(psych)

# carregando CSV de treino
dados_treino <- read.csv("train.csv", sep = ",", encoding = "UTF-8")

# visualizando dados
View(dados_treino)
str(dados_treino)
```

```r
# Feature selection
cols <- c("NU_NOTA_CN",
          "NU_NOTA_CH",
          "NU_NOTA_LC",
          "NU_NOTA_MT",
          "NU_NOTA_COMP1",
          "NU_NOTA_COMP2",
          "NU_NOTA_COMP5"
)

dados_treino_nm <-subset(dados_treino, select = cols)
```
```r
#Relação entre as variáveis escolhidas 
ggcorr(dados_treino_nm[, cols], label=T)
pairs.panels(dados_treino_nm[, cols])
```
```r
# Verificação dos valores NA
sapply(dados_treino_nm, function(x) sum(is.na(x)))
missmap(dados_treino_nm, main = "Valores Missing Observados")

#alterando valores NA para 0 
dados_treino_nm[is.na(dados_treino_nm)] <- 0
missmap(dados_treino_nm, main = "Valores Missing Observados")
```
```r
#gerando CSV de treino
write.csv(dados_treino_nm, "dados_treino_nm.csv", sep = ";", dec = ",",row.names = FALSE)
```

#          Criação do Modelo de Regressão Linear - Treino               

```r
#1° modelo usando regressao linear simples
model <- lm(NU_NOTA_MT ~ ., 
            data = dados_treino_nm)

summary(model)
```
```r
#2° modelo usando Random Forest
model2 <- randomForest(NU_NOTA_MT ~ ., 
                       data = dados_treino_nm)

model2
```
```r
# Salvando o modelo
saveRDS(model2, file = "lm_model.rds")

# Carregando o modelo
modelo <- readRDS("lm_model.rds")
```

# Carregando o Dataset de Teste para Predição 

```r
# carregando CSV de teste
dados_teste <- read.csv("test.csv", sep = ",", encoding = "UTF-8")

# visualizando dados
View(dados_teste)
summary(dados_teste)
```
```r
#selecionando as variaveis -target
cols2 <- c("NU_NOTA_CN",
          "NU_NOTA_CH",
          "NU_NOTA_LC",
          "NU_NOTA_COMP1",
          "NU_NOTA_COMP2",
          "NU_NOTA_COMP5"
)

dados_teste_nm <- subset(dados_teste, select = cols2)
```
```r
# verificação dos valores NA
sapply(dados_teste_nm, function(x) sum(is.na(x)))
missmap(dados_teste_nm, main = "Valores Missing Observados")
```
```r
#alterando valores NA para 0 
dados_teste_nm[is.na(dados_teste_nm)] <- 0
missmap(dados_teste_nm, main = "Valores Missing Observados")
```
```r
# reservando o NU_INSCRICAO 
answer <- data.frame(dados_teste$NU_INSCRICAO)
```

#   Realizando a Predição Utilizando o DF de Teste  

```r
pred <- predict(model2, newdata = dados_teste_nm)
View(pred)
```
```r
#salvando o resultado no data frame final
answer$NU_NOTA_MT = pred

colnames(answer)[1] <- 'NU_INSCRICAO'
colnames(answer)[2] <- 'NU_NOTA_MT'

# Substituindo valores negativos por zero
answer$NU_NOTA_MT <- ifelse(answer$NU_NOTA_MT < 0, 0, answer$NU_NOTA_MT)

View(answer)
```r
#gerando CSV 
# Gerando o CSV com as respostas preditas
write.csv(answer,
          'answer.csv', 
          row.names = FALSE,
          quote=FALSE)
```
```
