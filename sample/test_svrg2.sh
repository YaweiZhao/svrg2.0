
../bin/linsvrg \
train_x_fn=data/covtype-trn.x,\
train_target_fn=data/covtype-trn.lab,\
test_x_fn=data/covtype-tst.x,\
test_target_fn=data/covtype-tst.lab,\
loss=Logistic,\
svrg_interval=2,\
sgd_iterations=1,\
num_iterations=30,\
random_seed=1,\
learning_rate=0.1,\
lam=1e-5,\
test_interval=2,\
ShowLoss,\
ShowTiming,\
prediction_fn=output/covtype.prediction
