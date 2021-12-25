setwd("~/CursoPowerBI/Cap15")
getwd()

#Instalando os pacotes 
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape") 
install.packages("randomForest") 
install.packages("e1071")

#Carregando libs
library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

#Carregando dataframe
dados_clientes <- read.csv("dataset.csv")
View(dados_clientes)

#Análise Exploratoria, Limpeza e transformação

#Removendo a coluna ID
dados_clientes$ID <- NULL
View(dados_clientes)

#checando valores ausentes 
sapply(dados_clientes, function(x) sum (is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

#Renomeando colunas categóricas 
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridades"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"

colnames(dados_clientes)[24] <- "Inadimplente"

#Genero
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero, c(0,1,2), labels = c("masculino", "feminino"))
summary(dados_clientes$Genero)

#Escolaridades
summary(dados_clientes$Escolaridades)
dados_clientes$Escolaridades <- cut(dados_clientes$Escolaridades, 
                             c(0,1,2,3,4), 
                             labels = c("pos graduado", "graduado", "Ensino Medio", "Outros"))
summary(dados_clientes$Escolaridades)

#Estado Civil 
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil, 
                                    c(-1,0,1,2,3), 
                                    labels = c("Desconhecido",
                                               "Casado",
                                               "Solteiro",
                                               "Outro"))
summary(dados_clientes$Estado_Civil)
#idade
summary(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade, 
                                   c(0,30,50,100), 
                                   labels = c("Jovem",
                                              "Adulto",
                                              "Idoso"))
summary(dados_clientes$Idade)   

#Convertendo variaveis de pagamento para factor
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

#Após Conversão
str(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")

#Variável dependente para factor

dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)

#Verificando proporção de inadimplentes
table(dados_clientes$Inadimplente)

#porcentagem das classes
prop.table(table(dados_clientes$Inadimplente))

#plot de distribuição usando ggplot2
qplot(Inadimplente, data = dados_clientes, geom = "bar") + 
  theme(axis.text = element_text(angle = 90, hjust = 1))

#Set seed
set.seed(12345)

#Amostragem estratificada
#seleciona linhas d e acordo com a variável inadimplente como strata 
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)
dim(indice)

#Definimos os dados de treinamento como subconjunto do conjunto de dados original
#com números de indice (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$Inadimplente)

#verificar % entre as classes 
prop.table(table(dados_treino$Inadimplente))

#Camparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                       prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treianemnto", "Original")
compara_dados
#tudo que não esta no dataset de treinamento esta no dataset de teste. Utilizar o sinal -
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)

#Construindo primeira versão do modelo 
modelo_v1 <- randomForest(Inadimplente ~ ., data = dados_treino)
modelo_v1

#Avaliação do modelo 
plot(modelo_v1)

#previsões com dados teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Balanceamento de classe


# Download package tarball from CRAN archive

url <- "https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz"
pkgFile <- "DMwR_0.4.1.tar.gz"
download.file(url = url, destfile = pkgFile)

# Expand the zip file using whatever system functions are preferred

# look at the DESCRIPTION file in the expanded package directory

# Install dependencies list in the DESCRIPTION file

install.packages(c('xts', 'quantmod', 'zoo', 'abind', 'ROCR'))

# Install package
install.packages(pkgs=pkgFile, type="source", repos=NULL)

# Delete package tarball
unlink(pkgFile)

library(DMwR)

# Balanceamento de classe
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(Inadimplente ~ ., data  = dados_treino)                         
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))

# Construindo a segunda versão do modelo
modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive = "1")
cm_v2

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Importância das variáveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var), 
                            Importance = round(imp_var[ ,'MeanDecreaseGini'],2))
# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>% 
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importância relativa das variáveis
ggplot(rankImportance, 
       aes(x = reorder(Variables, Importance), 
           y = Importance, 
           fill = Importance)) + 
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), 
            hjust = 0, 
            vjust = 0.55, 
            size = 4, 
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

# Construindo a terceira versão do modelo apenas com as variáveis mais importantes
colnames(dados_treino_bal)
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, 
                          data = dados_treino_bal)
modelo_v3
# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
?caret::confusionMatrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive = "1")
cm_v3

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Salvando o modelo em disco
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")

# Novas previsões com novos dados de 3 clientes

# Dados dos clientes
PAY_0 <- c(0, 0, 0) 
PAY_2 <- c(0, 0, 0) 
PAY_3 <- c(1, 0, 0) 
PAY_AMT1 <- c(1100, 1000, 1200) 
PAY_AMT2 <- c(1500, 1300, 1150) 
PAY_5 <- c(0, 0, 0) 
BILL_AMT1 <- c(350, 420, 280) 

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

#Fazendo predições
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)

# Checando os tipos de dados
str(dados_treino_bal)
str(novos_clientes)


# Convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

#REfazendo predições
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
View(previsoes_novos_clientes)