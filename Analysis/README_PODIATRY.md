### Podiatry prediction steps

1. Remove sleep/idle times from raw data. Done
2. Rename the user ids - device ID to user id Participant001,... Done
2. Preprocess raw data into input format of the CNN model.
3. Predict using the trained model.
4. Save the outputs in the form - 
In 60-s epochs
x, y, z, Classification output (SB, LPA, MVPA), Regression output (MET value)