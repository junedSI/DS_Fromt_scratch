
class LinearRegression():
      def __init__(self,m,b,L,epoch):
            self.m = m
            self.b = b
            self.L = L
            self.epoch = epoch
      
      def loss_function(m, b, points):
            total_error = 0
            for i in range(len(points)):
                  x = points.iloc[i].studytime
                  y = points.iloc[i].score
                  total_error += (y-(m*x+b)) ** 2
            
            total_error/float(len(points))

      def gd(m_now, b_now, points, L):
           m_gradient = 0
           b_gradient = 0
           n = len(points)

           for i in range(n):
                 x = points.iloc[i].studytime
                 y = points.iloc[i].score

           m_gradient += -(2/n)*x*(y-(m_now*x+b_now))
           b_gradient += (2/n)*(y-(m_now*x+b_now))
           m = m_now - m_gradient * L
           b = b_now - b_gradient * L
           return m, b
      
      def _predict(self, X):
            predictions = []
            for sample in X:
                prediction = self.bias
                for i in range(len(sample)):
                    prediction += self.weights[i] * sample[i]
                predictions.append(prediction)
            return predictions
