import pandas, numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler


class Predict:
    data = None

    def __init__(self, csvfile='data.csv'):
        # user pandas to read the file
        dataframe = pandas.read_csv(csvfile, usecols=[1], engine='python', skipfooter=0)
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        self.data = dataset

    # convert an array of values into a dataset matrix
    def create_dataset(self, dataset, look_back=5):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 5):
            a = dataset[i:(i + look_back), 0]
            b = dataset[(i + look_back):(i + look_back + 5), 0]
            dataX.append(a)
            dataY.append(b)
        #dataX is the data for 5 days and dataY is the tag which is the next 5 days' data
        return numpy.array(dataX), numpy.array(dataY)



    def train_model(self):
        # transform the initial data into the number between (0,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(self.data)

        train_size = int(len(dataset))
        train = dataset[0:train_size, :]

        look_back = 5
        #get the numpy array trainX and it's tag trainY
        trainX, trainY = self.create_dataset(train, look_back)

        # reshape trainX to fit model.fit()
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        # train model and fit LSTM network
        model = Sequential()
        model.add(LSTM(6, input_dim=look_back))
        model.add(Dense(5))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
        return model


    def predict_new(self, input):
        model = self.train_model()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.data)
        inp = scaler.transform([input])
        #use model.predict() to predict and print out
        print(scaler.inverse_transform(model.predict(numpy.array(inp).reshape(1, 1, 5))))


def main():
    #the gold price from 4.28 to 5.5,using to predict the next 5 days price
    inptarr = [282.50,282.25,282.25,282.25,282.25]
    x = Predict()
    x.predict_new(inptarr)

if __name__ == "__main__":
    main()