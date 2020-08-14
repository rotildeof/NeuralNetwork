#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "GeneticAlgorithm.h"
#include "GeneticAlgorithm.cxx"

#ifndef _NeuralNetwork_
#define _NeuralNetwork_

using vec3D = std::vector<std::vector<std::vector<double> > >;
using vec2D = std::vector<std::vector<double> >;
using vec1D = std::vector<double>;
using vec1I = std::vector<int>;


class NeuralNetwork{
public:
  NeuralNetwork(std::string structure_);
  ~NeuralNetwork(){};
  

  void SetActivationFunction_Hidden(std::string actFuncHidden);
  void SetActivationFunction_Output(std::string actFuncOutput);
  void SetLossFunction(std::string nameLossFunction_);
  void SetLearningMethod(std::string method);
  //[i][j] : [i] --> Data entry, [j] --> ith Neuron in first (last) layer.
  void LearningGA(vec2D const &inputDataSet,
		vec2D const &answerDataSet,
		int nRepetitions);
  void LearningGA(vec2D const &inputDataSet,
		vec2D const &answerDataSet,
		double threshold);
  void TrainNN(vec2D const &inputDataSet,
	       vec2D const &answerDataSet,
	       int nRepetitions);
  void SetLossPrintFrequency(int freq){printFreq = freq;};
  void SetRangeOfWeight(double min, double max){lower = min; upper = max;};
  
  void SetPopulation(int population){population_ = population;};
  void SetMutationProbability(double prob){mutation_prob = prob;};
  void SetNumOfDominantGene(double nDominantGene_){nDominantGene = nDominantGene_;};
  
  void PrintWeightMatrix();
  void PrintLastLayer(vec2D const &inputDataSet);
  void PrintLastLayer(vec1D const &inputData);
  vec1D::iterator GetOutputIterator(vec1D const &inputData);
  vec1D::iterator GetLossIteratorBegin(){return losses.begin();};
  vec1D::iterator GetLossIteratorEnd(){return losses.end();};
  void SaveNeuralNetwork(std::string output_filename);
  void ReadWeightMatrix(std::string filename);

  
  private:
  std::string structure;
  // 3:3:2
  vec3D w;
  vec2D b;
  vec2D nLayerNeurons; 
  vec2D beforeActFunc; 
  vec2D delta;
  vec3D dw;
  vec2D db;

  vec1D losses;
  
  int numLastNeurons;
  void CalcuHiddenLayer();
  void CalculationAllStageOfLayer();
  void ParameterInitialization();
  double lower = -5.; // default
  double upper =  5.; // default
  template <class T> void InputData(std::vector<T> const &indata);
  static double Sigmoid(double x);
  static double ReLU(double x);
  static double Identity(double x);
  static double DSigmoid(double x);
  static double DReLU(double x);
  static double DIdentity(double x);
  static void Sigmoid(vec1D &lastLayer, vec1D const &beforeActF);
  static void Softmax(vec1D &lastLayer, vec1D const &beforeActF);
  static void Identity(vec1D &lastLayer, vec1D const &beforeActF);
  // static void DSigmoid(vec1D &outputVec, vec1D const &inputVec);
  // static void DSoftmax(vec1D &outputVec, vec1D const &inputVec);
  // static void DIdentity(vec1D &outputVec, vec1D const &inputVec);
  
  
  void CalculationLastLayerDelta(vec1D const &NNoutput, vec1D const &answer);
  void CalculationLastLayerDeltaSoftmax(vec1D const &NNoutput, vec1D const &answer);
  void CalculationAllLayerDelta();
  void CalcuGradientW();
  void SGD();
  void MomentumSGD();
  void Adam();
  double beta_1 = 0.9;      // Adam
  double beta_2 = 0.999;    // Adam
  double epsilon = 1e-8;    // Adam
  double alpha_Adam = 0.001;// Adam
  vec3D m_w;                // Adam
  vec3D v_w;                // Adam
  vec2D m_b;                // Adam
  vec2D v_b;                // Adam
  
  double (*hidf_ptr)(double x) = &NeuralNetwork::ReLU;  // "Sigmoid", "ReLU"
  void (*outf_ptr)(vec1D &lL, vec1D const &bAF) = &NeuralNetwork::Sigmoid;

  double (*hiddf_ptr)(double x) = &NeuralNetwork::DReLU;
  double (*outdf_ptr)(double x) = &NeuralNetwork::DSigmoid;
  void (NeuralNetwork::*deltaCalcu_ptr)(vec1D const &NNoutput, vec1D const &answer);
  void (NeuralNetwork::*method_ptr)();

  std::string funcNameHidden = "ReLU";
  std::string funcNameOutput = "Sigmoid";
  
  double learningRate = 0.01;
  double alpha = 0.9; // for momentum SGD
  
  vec3D dweight; // for momentum SGD
  vec2D doffset; // for momentum SGD

  int printFreq = 10000; // Default
  
  double population_ = 100; //default : for GA
  double mutation_prob = 0.05; // default : for GA
  double nDominantGene = 5; //default : for GA

  std::mt19937 mt;

  GeneticAlgorithm<double> GA;
  // ---- Loss Function ---- //
  double (*loss_ptr)(vec1D const &lastNeurons, vec1D const &answerData)
  = &NeuralNetwork::MeanSquaredError;
  double (*dloss_ptr)(double y, double d) = &NeuralNetwork::DMSE;
  static double MeanSquaredError(vec1D const &lastNeurons, vec1D const &answerData); // MSE
  static double BinaryCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData); // BCE
  static double CategoricalCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData); //CCE
  static double DMSE(double y, double d);
  static double DBCE(double y, double d);
  static double DCCE(double y, double d);
  // ---------------------- //
  void SetWeightFromGene(GeneticAlgorithm<double> &GA, int ith_creature);
  void ShowInputAndOutput(GeneticAlgorithm<double> &GA, int ith_creature,
			  vec2D const &input);
  
  double LIMITLESS_1 = 0.9999999999999999;
};

#endif
