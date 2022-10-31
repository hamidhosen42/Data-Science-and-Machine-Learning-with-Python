from sklearn.metrics import mean_squared_error, r2_score

# Predicting the Test set results
pred=reg.predict(x_test)
pred

mean_squared_error(y_test, pred)

r2_score(y_test,pred)
