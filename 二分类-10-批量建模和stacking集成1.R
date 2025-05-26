library(stacks)
# 也可以是之前单个模型建模的结果
load(".\\cls2\\evalresult_SVM.RData")
load(".\\cls2\\evalresult_xgboost.RData")
load(".\\cls2\\evalresult_rf.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(tune_dt) %>%
  add_candidates(tune_rf) %>%
  add_candidates(cv_logistic)
models_stack

##############################

# 拟合stacking元模型——lasso
set.seed(42)
meta_stack <- blend_predictions(
  models_stack, 
  penalty = 10^seq(-2, -0.5, length = 20)
)
meta_stack
autoplot(meta_stack)

# 拟合选定的基础模型
set.seed(42)
final_stack <- fit_members(meta_stack)
final_stack

######################################################

# 以下遭遇 lightgbm 会报错
# 应用stacking模型预测并评估

# 训练集
predtrain_stack <- eval4cls2(
  model = final_stack, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "stacking", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$predprobplot_fill
predtrain_stack$rocresult
predtrain_stack$rocplot
predtrain_stack$prresult
predtrain_stack$prplot
predtrain_stack$cmresult
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff
predtrain_stack$ksplot
predtrain_stack$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_stack$proc)
pROC::ci.auc(predtrain_stack$proc)

# 测试集
predtest_stack <- eval4cls2(
  model = final_stack, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "stacking", 
  datasetname = "testdata",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_stack$prediction
predtest_stack$predprobplot
predtest_stack$predprobplot_fill
predtest_stack$rocresult
predtest_stack$rocplot
predtest_stack$prresult
predtest_stack$prplot
predtest_stack$cmresult
predtest_stack$cmplot
predtest_stack$metrics
predtest_stack$diycutoff
predtest_stack$ksplot
predtest_stack$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_stack$proc)
pROC::ci.auc(predtest_stack$proc)

# ROC比较检验
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)

# 合并训练集和测试集上ROC曲线
predtrain_stack$rocresult %>%
  bind_rows(predtest_stack$rocresult) %>%
  mutate(dataAUC = paste(data, " ROCAUC:", round(ROCAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = 1-specificity,
             y = sensitivity, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(1,0),
        legend.justification = c(1,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上PR曲线
predtrain_stack$prresult %>%
  bind_rows(predtest_stack$prresult) %>%
  mutate(dataAUC = paste(data, " PRAUC:", round(PRAUC, 4)),
         dataAUC = forcats::as_factor(dataAUC)) %>%
  ggplot(aes(x = recall,
             y = precision, 
             color = dataAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", slope = -1, intercept = 1) +
  scale_x_continuous(expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0), limits = c(0, 1)) +
  labs(color = "") +
  theme_bw() +
  theme(legend.position = c(0,0),
        legend.justification = c(0,0),
        legend.background = element_blank(),
        legend.key = element_blank())

# 合并训练集和测试集上性能指标
predtrain_stack$metrics %>%
  bind_rows(predtest_stack$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)


# 保存评估结果
save(predtrain_stack,
     predtest_stack,
     file = ".\\cls2\\evalresult_stack.RData")

###################################################################
######################## DALEX解释对象
# 自变量数据集
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-Early_recurrence)
colnames(traindatax)

explainer_stack <- DALEXtra::explain_tidymodels(
  final_stack, 
  data = traindatax,
  y = ifelse(traindata$Early_recurrence == yourpositivelevel, 1, 0),
  type = "classification",
  label = "stacking"
)
# 变量重要性
set.seed(42)
vip_stack <- DALEX::model_parts(
  explainer_stack,
  type = "ratio"
)
# plot(vip_stack) +
#   labs(subtitle = NULL)
plot(vip_stack, show_boxplots = FALSE) +
  labs(subtitle = NULL)

# 变量偏依赖图
# 连续型变量
set.seed(42)
pdpc_stack <- DALEX::model_profile(
  explainer_stack,
  variables = colnames(traindatax)
)
plot(pdpc_stack) +
  labs(subtitle = NULL)
# 分类变量
set.seed(42)
pdpd_stack <- DALEX::model_profile(
  explainer_stack,
  variables = colnames(traindatax)[c(2,3,6,7,9,11,13)]  # 分类变量所在位置
)
plot(pdpd_stack) +
  labs(subtitle = NULL)

# 单样本预测分解
set.seed(42)
shap_stack <- DALEX::predict_parts(
  explainer = explainer_stack, 
  new_observation = traindatax[2, ], 
  type = "shap"
)
# plot(shap_stack, 
#      max_features = ncol(traindatax))
plot(shap_stack, 
     max_features = ncol(traindatax),
     show_boxplots = FALSE)
shapviz::sv_force(shapviz::shapviz(shap_stack))
shapviz::sv_waterfall(shapviz::shapviz(shap_stack))

######################## fastshap包

shapresult <- shap4cls2(
  finalmodel = final_stack,
  predfunc = function(model, newdata) {
    predict(model, newdata, type = "prob") %>%
      select(ends_with(yourpositivelevel)) %>%
      pull()
  },
  datax = traindatax,
  datay = traindata$Early_recurrence,
  yname = "Early_recurrence",
  flname = colnames(traindatax)[c(2,3,6,7,9,11,13)],
  lxname = colnames(traindatax)[-c(2,3,6,7,9,11,13)]
)

# 基于shap的变量重要性
shapresult$shapvip
shapresult$shapvipplot
# 单样本预测分解
# 1 采用shapviz包
shapley <- shapviz::shapviz(
  shapresult$shapley,
  X = traindatax,
  baseline = mean(predtrain_stack$prediction$.pred_Yes)
)
shapviz::sv_force(shapley, row_id = 1)  # 第1个样本
shapviz::sv_waterfall(shapley, row_id = 1)  # 第1个样本

shapresult$shapplotd_facet
shapresult$shapplotd_one
# 所有连续变量的shap图示
shapresult$shapplotc_facet
shapresult$shapplotc_one
shapresult$shapplotc_one2
# 单分类变量shap图示
shap1d <- shapresult$shapdatad %>%
  dplyr::filter(feature == "ChestPain") %>% # 某个要展示的分类自变量
  na.omit() %>%
  ggplot(aes(x = value, y = shap)) +
  geom_boxplot(fill = "lightgreen") +
  geom_point(aes(color = Y), alpha = 0.5) + 
  geom_hline(yintercept = 0, color = "grey10") +
  scale_color_viridis_d() +
  labs(x = "ChestPain", color = "Early_recurrence", y = "Shap") + # 自变量名称和因变量名称
  theme_bw()
shap1d
ggExtra::ggMarginal(
  shap1d + theme(legend.position = "bottom"),
  type = "histogram",
  margins = "y",
  fill = "skyblue"
)

# 单连续变量shap图示
shap1c <- shapresult$shapdatac %>%
  dplyr::filter(feature == "Age") %>% # 某个要展示的连续自变量
  na.omit() %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point(aes(color = Y)) +
  geom_smooth(color = "red") +
  geom_hline(yintercept = 0, color = "grey10") +
  scale_color_viridis_d() +
  labs(x = "Age", color = "Early_recurrence", y = "Shap") + # 自变量名称和因变量名称
  theme_bw()
shap1c
ggExtra::ggMarginal(
  shap1c + theme(legend.position = "bottom"),
  type = "histogram",
  fill = "skyblue"
)
