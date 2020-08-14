#include <iostream>
#include "NeuralNetwork.h"
#include "NeuralNetwork.cxx"
#include <cmath>
#include <vector>
#include <random>

int main(){

  double pi = 3.14159265358979323;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> uni(0, 3);
  //std::uniform_int_distribution<> uni(0, 1);
  int nData = 4000;
  std::vector<std::vector<double> > input(nData);
  std::vector<std::vector<double> > answer(nData);

  for(int i = 0 ; i < nData ; i++){
    double x = uni(mt);
    double y = std::floor(x);

    input[i].push_back(x);
    answer[i].push_back(y);
  }
  
  NeuralNetwork NN("1:10:10:10:10:1");
  NN.SetRangeOfWeight(-5.0, 5.0);
  NN.SetActivationFunction_Hidden("Sigmoid");
  NN.SetActivationFunction_Output("Identity");
  NN.SetLossFunction("MSE");
  NN.SetLearningMethod("Adam");
  NN.TrainNN(input, answer, 2000000);
  NN.PrintWeightMatrix();

  std::cout << std::endl;
  NN.SaveNeuralNetwork("test");

return 0;
}
