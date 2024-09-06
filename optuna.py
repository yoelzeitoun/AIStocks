# import optuna
#
# class Optuna:
#     def __init__(self):
#         print("OPTUNA")
#
#     def objective(trial):
#         pass
#       # Define parameters to optimize
#       # filters = trial.suggest_int('filters', 4, 64, step=4)
#       # kernel_size = trial.suggest_int('kernel_size', 3, 7)
#       # learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-1)
#       # optimizer_choice = trial.suggest_categorical('optimizer', ['Nadam', 'Adam', 'RMSprop'])
#       # activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid'])
#
#       # Set optimizer based on choice
#       # if optimizer_choice == 'Nadam':
#       #     optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
#       # elif optimizer_choice == 'Adam':
#       #     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#       # elif optimizer_choice == 'RMSprop':
#       #     optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
#
#       # Build model
#       # input_shape = (x_train.shape[1], x_train.shape[2])
#       # model = Sequential()
#       # model.add(GRU(filters, dropout=dropout_rate, recurrent_dropout=dropout_rate_recurrent, return_sequences=True, input_shape=input_shape))
#
#       # model.add(GRU(int(filters/2), dropout=dropout_rate, recurrent_dropout=dropout_rate_recurrent, input_shape=input_shape))
#       # model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=activation, input_shape=input_shape))
#       # model.add(MaxPooling1D(3))
#       # model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
#       # model.add(GlobalMaxPooling1D())
#       # model.add(Dense(1, activation='sigmoid'))  # Binary output
#       # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#
#       # Train model
#       # history = model.fit(x_train, y_train_binary, epochs=100, batch_size=32, validation_data=(x_val, y_val_binary), callbacks=[early_stopping], verbose=0)
#       # y_pred_binary = test_model(model, x_test, y_test_binary)
#       # max_consecutive_y_pred = max_consecutive(y_pred_binary)
#       # max_consecutive_y_test = max_consecutive(y_test_binary)
#       # Evaluate distribution using chi-square test
#       # if is_distribution_normal(y_pred_binary) and (max_consecutive_y_pred < 3*max_consecutive_y_test):
#       #   return 0.0  # Return 0.0 for perfect match
#       # else:
#       #   return 1.0  # Penalize non-50-50 distributions
#
#     def optimize(self):
#         # Create study and optimize
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=10)
#
#         # Retrieve best parameters
#         best_params = study.best_params
#         print("Best parameters:", best_params)
