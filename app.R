library(shapviz)
library(shiny)
library(shinythemes)
library(tidymodels)
library(stacks)
library(lightgbm)
library(rsconnect)
library(bslib)
library(bonsai)
library(randomForest)
library(kernlab)
library(xgboost)
library(shinyjs)
library(DALEX)
library(iBreakDown)

# 加载数据 - 确保路径正确
load("shiny_stack_heart.RData")

# 转换目标变量
traindata_heart$Early_recurrence <- as.numeric(traindata_heart$Early_recurrence) - 1

# 自定义预测函数 - 这是解决DALEX问题的关键
stack_predict <- function(model, newdata) {
  pred <- predict(model, new_data = newdata, type = "prob")
  return(pred$.pred_1)
}

# 创建解释对象
explainer_stack_heart <- explain(
  model = final_stack_heart,
  data = traindata_heart[, -which(names(traindata_heart) == "Early_recurrence")],
  y = traindata_heart$Early_recurrence,
  predict_function = stack_predict,  # 使用自定义预测函数
  type = "classification",
  label = "Final Stacked Model"
)

# UI部分保持不变
ui <- fluidPage(
  theme = bs_theme(version = 4, bootswatch = "flatly"),
  tags$head(
    tags$style(HTML("
      /* General styling */
      body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
      }
      .shiny-input-container {
        margin-bottom: 20px;
      }
      .well {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
      }
      .well h4 {
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 15px;
      }
      .action-button {
        width: 100%;
        margin-top: 20px;
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
      }
      .action-button:hover {
        background-color: #2980b9;
      }
      .results-panel {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
      }
      .results-panel h3 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        font-weight: bold;
      }
      .disclaimer {
        font-size: 0.9em;
        color: #7f8c8d;
        margin-top: 15px;
      }
      .numeric-input, .select-input {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 8px;
        width: 100%;
      }
      .selectize-input {
        border-radius: 5px;
        border: 1px solid #ddd;
      }
      .info-text {
        font-size: 0.8em;
        color: #7f8c8d;
      }
      .info-icon {
        color: #3498db;
        cursor: pointer;
      }
    "))
  ),
  
  titlePanel(div("NSCLC Early Recurrence Prediction", 
                 style = "color: #2c3e50; text-align: center; font-weight: bold; font-size: 28px; margin-bottom: 20px;")),
  
  fluidRow(
    column(8,
           fluidRow(
             column(4,
                    wellPanel(
                      selectInput("pT", "pT", choices = c("T1" = 1, "T2a" = 2, "T2b" = 3, "T3" = 4, "T4" = 5), selected = 3),
                      div(class = "info-text", "Pathological tumor (pT) stage, indicating the size and extent of the primary tumor based on surgical pathology."),
                      numericInput("Maximal_diameter", "Maximal diameter (cm)", min = 0, max = 15, value = 5, step = 0.1),
                      div(class = "info-text", "The maximum diameter of the tumor should be determined based on the pathological report."),
                      numericInput("CA125", "CA125 (U/mL)", value = 40, min = 0, max = 400),
                      div(class = "info-text", "Preoperatively measured serum level of the CA125 tumor marker."),
                      numericInput("CYFRA21_1", "CYFRA21-1 (ng/ml)", value = 3.5, min = 0, max = 800),
                      div(class = "info-text", "Preoperatively measured serum level of the CYFRA21-1 tumor marker."),
                      selectInput("pN", "pN", choices = c("N0" = 0, "N1" = 1, "N2" = 2), selected = 1),
                      div(class = "info-text", "Pathological nodal (pN) stage, reflecting lymph node metastasis confirmed by histopathological analysis."),
                      selectInput("Tumor_site", "Tumor site", 
                                  choices = c("Right lower lobe" = 1, "Right middle lobe" = 2, "Right upper lobe" = 3, "Left lower lobe" = 4, "Left upper lobe" = 5), selected = 1),
                      div(class = "info-text", "Primary site of the lung cancer, as confirmed by histopathology."),
                      selectInput("Degree_of_differentiation", "Degree of differentiation", 
                                  choices = c("Well" = 1, "Moderate" = 2, "Poor" = 3, "Undifferentiated" = 4), selected = 2),
                      div(class = "info-text", "The degree to which tumor cells resemble normal cells.")
                    )
             ),
             column(4,
                    wellPanel(
                      selectInput("Extent_of_resection", "Extent of resection", 
                                  choices = c("Segmentectomy" = 1, "Lobectomy" = 2, "Pneumonectomy" = 3), selected = 2),
                      div(class = "info-text", "Extent of surgical removal of the tumor."),
                      selectInput("Lymphovascular_invasion", "Lymphovascular invasion", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "Indicates whether cancer cells have spread to lymphatic or blood vessels(postoperative specimen)."),
                      selectInput("Clavien_Dindo", "Clavien-Dindo", 
                                  choices = c("Grade 0" = 0, "Grade 1" = 1, "Grade 2" = 2, "Grade 3" = 3, "Grade 4" = 4), selected = 0),
                      div(class = "info-text", "Classification of surgical complications."),
                      selectInput("Visceral_pleural_invasion", "Visceral pleural invasion", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "Pathological confirmation of pleural invasion (postoperative specimen)."),
                      numericInput("CEA", "CEA (ng/ml)", value = 5, min = 0, max = 850),
                      div(class = "info-text", "Preoperatively measured serum level of the CEA tumor marker."),
                      numericInput("Hb", "Hemoglobin (g/L)", value = 120, min = 30, max = 200),
                      div(class = "info-text", "Preoperative hemoglobin level (from routine blood test)"),
                      selectInput("Pathological_type", "Pathological type", 
                                  choices = c("Adenocarcinoma" = 1, 
                                              "Squamous cell carcinoma" = 2, 
                                              "Adenosquamous carcinoma" = 3, 
                                              "Other" = 4), 
                                  selected = 2),
                      div(class = "info-text", "Pathological classification of the tumor type.")
                    )
             ),
             column(4,
                    wellPanel(
                      numericInput("ALB", "ALB (g/L)", value = 40, min = 20, max = 60),
                      div(class = "info-text", "Preoperatively measured serum level of albumin, a protein in the blood."),
                      selectInput("Adjuvant_chemotherapy", "Adjuvant chemotherapy", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "Indicates whether the patient received additional chemotherapy after surgery."),
                      selectInput("PNI", "Prognostic nutritional index", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "Indicates whether cancer has invaded the nerves."),
                      selectInput("Neoadjuvant", "Neoadjuvant", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "Neoadjuvant therapy status for lung cancer (preoperative)."),
                      selectInput("TP53", "TP53", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "TP53 mutation status as detected by molecular testing of the resected tumor specimen."),
                      selectInput("Pulmonary_complications", "Pulmonary complications", choices = c("No" = 0, "Yes" = 1), selected = 0),
                      div(class = "info-text", "Indicates whether the patient has experienced any pulmonary complications."),
                      actionButton("predict", "Predict", class = "btn-primary action-button")
                    )
             )
           )
    ),
    column(4,
           div(class = "results-panel",
               h3("Results"),
               verbatimTextOutput("probability"),
               verbatimTextOutput("risk"),
               h3("SHAP Explanation"),
               plotOutput("shapley"),
               h3("Disclaimer"),
               div(class = "disclaimer",
                   uiOutput("disclaimer")
               )
           )
    )
  )
)

# Server部分
server <- function(input, output) {
  # 使用全局环境中的数据和模型，避免重复加载
  
  # 创建新数据反应式
  new_data <- reactive({
    # 确保因子水平与训练数据一致
    data.frame(
      pN = factor(input$pN, levels = c(0, 1, 2)),
      pT = factor(input$pT, levels = 1:5),
      Lymphovascular_invasion = factor(input$Lymphovascular_invasion, levels = c(0, 1)),
      Maximal_diameter = input$Maximal_diameter,
      CA125 = input$CA125,
      CYFRA21_1 = input$CYFRA21_1,
      CEA = input$CEA,
      ALB = input$ALB,
      Hb = input$Hb,
      Adjuvant_chemotherapy = factor(input$Adjuvant_chemotherapy, levels = c(0, 1)),
      Visceral_pleural_invasion = factor(input$Visceral_pleural_invasion, levels = c(0, 1)),
      Pathological_type = factor(input$Pathological_type, levels = 1:4),
      Degree_of_differentiation = factor(input$Degree_of_differentiation, levels = 1:4),
      Tumor_site = factor(input$Tumor_site, levels = 1:5),
      Extent_of_resection = factor(input$Extent_of_resection, levels = 1:3),
      Neoadjuvant = factor(input$Neoadjuvant, levels = c(0, 1)),
      TP53 = factor(input$TP53, levels = c(0, 1)),
      Clavien_Dindo = factor(input$Clavien_Dindo, levels = 0:4),
      Pulmonary_complications = factor(input$Pulmonary_complications, levels = c(0, 1)),
      PNI = factor(input$PNI, levels = c(0, 1))
    )
  })
  
  # 预测结果
  prediction <- eventReactive(input$predict, {
    pred <- predict(final_stack_heart, new_data = new_data(), type = "prob")
    prob <- pred$.pred_1
    list(probability = prob, risk_level = ifelse(prob > 0.171, "high risk", "low risk"))
  })
  
  output$probability <- renderText({
    req(prediction())
    paste("Early recurrence probability:", sprintf("%.1f%%", prediction()$probability * 100))
  })
  
  output$risk <- renderText({
    req(prediction())
    paste("Risk level:", prediction()$risk_level)
  })
  
  # SHAP分解图
  output$shapley <- renderPlot({
    req(input$predict)  # 只在点击预测按钮后显示
    
    # 使用iBreakDown包的shap函数
    shap_explanation <- iBreakDown::shap(
      explainer_stack_heart,
      new_observation = new_data()
    )
    
    # 绘制SHAP解释图
    plot(shap_explanation) +
      ggtitle("SHAP Values for Prediction") +
      theme_minimal()
  })
  
  # Disclaimer输出
  output$disclaimer <- renderUI({
    req(prediction())
    tagList(
      p("This application is intended for educational and informational purposes only. The information provided by this application is not a substitute for professional medical advice, diagnosis, or treatment."),
      p("The content generated by this application should not be used as the sole basis for making medical decisions. Users should consult with a qualified healthcare provider before making any medical decisions or if they have any questions about a medical condition."),
      p("We do not assume any liability for the use of this application. The accuracy, completeness, and timeliness of the information provided cannot be guaranteed."),
      p("By using this application, you acknowledge and agree that you are doing so at your own risk.")
    )
  })
}

# 运行应用
shinyApp(ui = ui, server = server)
