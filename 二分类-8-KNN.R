# 数据预处理配方
datarecipe_knn <- recipe(Early_recurrence ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), 
             naming = new_dummy_names) %>% 
  step_normalize(all_predictors())
datarecipe_knn


# 设定模型
model_knn <- nearest_neighbor(
  mode = "classification",
  engine = "kknn",
  
  neighbors = tune(),
  weight_func = tune(),
  dist_power = 2
)
model_knn

# workflow
wk_knn <- 
  workflow() %>%
  add_recipe(datarecipe_knn) %>%
  add_model(model_knn)
wk_knn

##############################################################

############################  超参数寻优1-网格搜索

# 超参数寻优网格
set.seed(42)
hpgrid_knn <- parameters(
  neighbors(range = c(3, 11)),
  weight_func()
) %>%
  # grid_regular(levels = c(5)) # 常规网格
  grid_random(size = 20) # 随机网格
# grid_latin_hypercube(size = 10) # 拉丁方网格
# grid_max_entropy(size = 10) # 最大熵网格
hpgrid_knn
# 网格也可以自己手动生成expand.grid()
# 交叉验证网格搜索过程
set.seed(42)
tune_knn <- wk_knn %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_knn,
    metrics = metric_set(yardstick::accuracy, 
                         yardstick::roc_auc, 
                         yardstick::pr_auc),
    control = control_grid(save_pred = T, 
                           verbose = T,
                           event_level = "second",
                           parallel_over = "everything",
                           save_workflow = T)
  )

#########################  超参数寻优2-贝叶斯优化
library(kknn)

# 贝叶斯优化超参数
set.seed(42)
tune_knn <- wk_knn %>%
  tune_bayes(
    resamples = folds,
    initial = 20,
    iter = 50,
    metrics = metric_set(yardstick::roc_auc,
                         yardstick::accuracy, 
                         yardstick::pr_auc),
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )

########################  超参数寻优结束

# 交叉验证结果
eval_tune_knn <- tune_knn %>%
  collect_metrics()
eval_tune_knn

# 图示
autoplot(tune_knn)
eval_tune_knn %>% 
  filter(.metric == "roc_auc") %>%
  mutate(weight_func2 = as.numeric(as.factor(weight_func))) %>%
  plotly::plot_ly(
    type = 'parcoords',
    line = list(color = ~mean, colorscale = 'Jet', showscale = T),
    dimensions = list(
      list(label = 'neighbors', values = ~neighbors),
      list(label = 'weight_func', values = ~weight_func2,
           range = c(1,length(unique(eval_tune_knn$weight_func))), 
           tickvals = 1:length(unique(eval_tune_knn$weight_func)),
           ticktext = sort(unique(eval_tune_knn$weight_func)))
    )
  ) %>%
  plotly::layout(title = "KNN HPO Guided by AUCROC")

# 经过交叉验证得到的最优超参数
hpbest_knn <- tune_knn %>%
  select_by_one_std_err(metric = "roc_auc", desc(neighbors))
hpbest_knn
"# A tibble: 1 × 3
  neighbors weight_func .config              
      <int> <chr>       <chr>                
1        15 rectangular Preprocessor1_Model08"
# 采用最优超参数组合训练最终模型
set.seed(42)
final_knn <- wk_knn %>%
  finalize_workflow(hpbest_knn) %>%
  fit(traindata)
final_knn
# 保存模型
saveRDS(final_knn, file = "final_knn_model.rds")
# 读取保存的模型
# final_knn <- readRDS("final_knn_model.rds")
##################################################################

# 训练集预测评估
predtrain_knn <- eval4cls2(
  model = final_knn, 
  dataset = traindata, 
  yname = "Early_recurrence", 
  modelname = "KNN", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_knn$prediction
predtrain_knn$predprobplot
predtrain_knn$predprobplot_fill
predtrain_knn$rocresult
predtrain_knn$rocplot
predtrain_knn$prresult
predtrain_knn$prplot
predtrain_knn$cmresult
predtrain_knn$cmplot
predtrain_knn$metrics
predtrain_knn$diycutoff
predtrain_knn$ksplot
predtrain_knn$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtrain_knn$proc)
pROC::ci.auc(predtrain_knn$proc)

# 预测评估测试集预测评估
predtest_knn <- eval4cls2(
  model = final_knn, 
  dataset = testdata, 
  yname = "Early_recurrence", 
  modelname = "KNN", 
  datasetname = "testdata",
  cutoff = predtrain_knn$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_knn$prediction
predtest_knn$predprobplot
predtest_knn$predprobplot_fill
predtest_knn$rocresult
predtest_knn$rocplot
predtest_knn$prresult
predtest_knn$prplot
predtest_knn$cmresult
predtest_knn$cmplot
predtest_knn$metrics
predtest_knn$diycutoff
predtest_knn$ksplot
predtest_knn$dcaplot

# pROC包auc值及其置信区间
pROC::auc(predtest_knn$proc)
pROC::ci.auc(predtest_knn$proc)

# ROC比较检验
pROC::roc.test(predtrain_knn$proc, predtest_knn$proc)

# 合并训练集和测试集上ROC曲线
predtrain_knn$rocresult %>%
  bind_rows(predtest_knn$rocresult) %>%
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
predtrain_knn$prresult %>%
  bind_rows(predtest_knn$prresult) %>%
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
predtrain_knn$metrics %>%
  bind_rows(predtest_knn$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
# 合并训练集和测试集上的性能指标
performance_metrics <- predtrain_knn$metrics %>%
  bind_rows(predtest_knn$metrics) %>%
  select(-.estimator) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# 最优超参数交叉验证的结果
evalcv_knn <- bestcv4cls2(
  wkflow = wk_knn,
  tuneresult = tune_knn,
  hpbest = hpbest_knn,
  yname = "Early_recurrence",
  modelname = "KNN",
  v = 5,
  positivelevel = yourpositivelevel
)
evalcv_knn$plotcv
evalcv_knn$evalcv

# 保存评估结果
save(datarecipe_knn,
     model_knn,
     wk_knn,
     # hpgrid_knn,
     tune_knn,
     predtrain_knn,
     predtest_knn,
     evalcv_knn,
     file = ".\\cls2\\evalresult_knn.RData")