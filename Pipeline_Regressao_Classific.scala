// Databricks notebook source
// Pipeline de Regressão Linear com Scala e Spark

// Módulos
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

// Definindo o nível do log
import  org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Inicializando a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Carregando os dados
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("dbfs:/FileStore/shared_uploads/eric.passos@dataside.com.br/dataset1-5.csv")

// Verificando os dados
data.printSchema()

// Separando as colunas e a primeira linha
val colnames = data.columns
val firstrow = data.head(1)(0)

// Imprimindo a primeira linha do dataset
println("\n")
println("Linha do dataset")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

// Configurando um Dataframe

// Precisamos definir o dataset na forma de duas colunas ("label", "features")
// Isso nos permitirá juntar várias colunas em uma única coluna de uma matriz de valores. 

// Imports
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector

// Criando o dataframe
val df =data.select(data("Valor").as("label"),$"Media_Salarial_Vizinhanca",
                    $"Media_Idade_Casa",$"Media_Numero_Comodos",$"Media_Numero_Quartos",$"Populacao_Vizinhanca")

// Um assembler converte os valores de entrada em um vetor
// Um vetor é o que o algoritmo ML lê para treinar o modelo

// Define as colunas de entrada das quais devemos ler os valores
// Define o nome da coluna onde o vetor será armazenado
val assembler = new VectorAssembler().setInputCols(Array("Media_Salarial_Vizinhanca",
                                                         "Media_Idade_Casa",
                                                         "Media_Numero_Comodos", "Media_Numero_Quartos",
                                                         "Populacao_Vizinhanca")).setOutputCol("features")

// Transforma o dataset em um objeto de duas colunas, no formato esperado pelo modelo
val output = assembler.transform(df).select($"label",$"features")

// Imprimindo a versão final do dataframe que vai alimentar o modelo de regressão
output.show()


// Configurando o modelo de regressão

// Criar um modelo de RegressãO liner 
val lr = new LinearRegression()

// Grid de Hiperparâmetros
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,
                                               Array (0.1, 0.01)).addGrid(lr.fitIntercept).addGrid(lr.elasticNetParam,
                                                                                                               Array(0.0, 0.5, 1.0)).build()

// Divide em dados de treino e teste
val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new                                                                              RegressionEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)

// Fit do modelo nos dados
val lrModel = lr.fit(output)

// Imprimir os coeficientes aprendidos no treinamento do modelo de regressão linear
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


// Avaliação
// Resumindo o Modelo
val trainingSummary = lrModel.summary

// Resíduos e Previsões
trainingSummary.residuals.show()
trainingSummary.predictions.show()

// Métricas
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"R2: ${trainingSummary.r2}")

// COMMAND ----------

// Pipilene de Classificação com Regressão Logística
// Prevendo se um passageiro vai sobreviver ou não ao naufrágio de um navio

// Módulos
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

// Definindo o nível de informação no log (nese caso, Erro)
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Spark Session
val spark = SparkSession.builder().getOrCreate()

// Carregando o dataset
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("dbfs:/FileStore/shared_uploads/eric.passos@dataside.com.br/dataset2-1.csv")

// Print do Schema do dataframe
data.printSchema()

// Visualizando os dados
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println(("Linha de exemplo do dataframe"))
for(ind <- Range(1, colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

// Feature Engineering 
// Data Wragling - Manipulando o dataset para o modelo preditivo

// Obtendo apenas as colunas necessárias para o modelo
val logregdataall = data.select(data("Survived").as("label"), $"Pclass", $"Sex", $"Age", $"SibSp", $"Parch", $"Fare", $"Embarked")

// Removendo linhas com valores NA
val logregdata = logregdataall.na.drop()

// Precisamos lidar com as colunas categóricas

// Import dos módulos VectorAssembler e Vectors
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Convertendo strings em valores numéricos (label_enconding)
val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

// Convertendo valores numéricos em One-Hot Encoding 0 ou 1
val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

// Montando o dataset  para o formato ("label", "features")
val assembler = (new VectorAssembler().setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkVec")).setOutputCol("features"))

// Dataset de treino e de teste
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

// Import do módulo
import org.apache.spark.ml.Pipeline

// Criando o objeto
val lr = new LogisticRegression()

// Criando o Pipeline
val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, lr))

// Construindo o Modelo, Métricas de Avaliação e Confusion Matrix

// Fit do pipeline nos dados de treino
val model = pipeline.fit(training)

// Obtendo resultados no dataset de Teste
val results = model.transform(test)

// Módulo para métricas de avaliação
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Precisamos converter para um RDD
val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd

// Instanciando as métricas do objeto
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion Matrix
println("Confusion Matrix:")
println(metrics.confusionMatrix)
println("Acurácia:")
println(metrics.accuracy)
