#include "NeuralNetwork.h"
#include <iomanip>
#include <fstream>
#include <sstream>

#define DBPR

NeuralNetwork::NeuralNetwork(std::string structure_){
  structure = structure_;
  std::string str = structure + ":";
  vec1I neurons;
  std::string store;
  for(int i = 0 ; i < (int)str.size() ; i++){
    if(str[i] != ':'){
      store += str[i];
    }else{
      int num = atoi(store.data());
      neurons.push_back(num);
      store.clear();
    }
  }
  w.assign(neurons.size() - 1, vec2D(0, vec1D() ) );
  b.assign(neurons.size() - 1, vec1D() );
  dw.assign(neurons.size() - 1, vec2D(0, vec1D() ) );
  db.assign(neurons.size() - 1, vec1D() );
  dweight.assign(neurons.size() - 1, vec2D(0, vec1D() ) );
  doffset.assign(neurons.size() - 1, vec1D() );
  m_w.assign(neurons.size() - 1, vec2D(0, vec1D() ) ); // Adam
  v_w.assign(neurons.size() - 1, vec2D(0, vec1D() ) ); // Adam
  m_b.assign(neurons.size() - 1, vec1D() ); // Adam
  v_b.assign(neurons.size() - 1, vec1D() ); // Adam
  std::cout << "NN Structure --> " << structure << "  Number of Layers --> " << neurons.size() << std::endl;
  for(int i = 0 ; i < (int)w.size() ; i++){
    w[i].assign(neurons[i+1], vec1D(neurons[i]) ); // weight
    b[i].assign(neurons[i+1],0);                   // offset
    dw[i].assign(neurons[i+1], vec1D(neurons[i]) ); // dE/dw
    db[i].assign(neurons[i+1],0);                   // dE/db
    dweight[i].assign(neurons[i+1], vec1D(neurons[i]) ); // Momentum SGD
    doffset[i].assign(neurons[i+1], 0 );                 // Momentum SGD
    m_w[i].assign(neurons[i+1], vec1D(neurons[i]) ); // Adam
    v_w[i].assign(neurons[i+1], vec1D(neurons[i]) ); // Adam
    m_b[i].assign(neurons[i+1],0);                   // Adam
    v_b[i].assign(neurons[i+1],0);                   // Adam
    std::cout << i << " : " << "Weight Matrix --> [" << neurons[i+1] << "][" << neurons[i] << "]  Offset Vector --> [" << neurons[i+1] << "]" << std::endl;
  }
  
  nLayerNeurons.assign(neurons.size(), vec1D());
  beforeActFunc.assign(neurons.size(), vec1D());
  delta        .assign(neurons.size(), vec1D());
  for(int i = 0 ; i < (int)nLayerNeurons.size() ; i++){
    nLayerNeurons[i].assign(neurons[i], 0);
    beforeActFunc[i].assign(neurons[i], 0); // nLayerZ[0][i] = 0;
    delta        [i].assign(neurons[i], 0); // delta for back propagation
  }

  numLastNeurons = *(neurons.end() -1);

  deltaCalcu_ptr = &NeuralNetwork::CalculationLastLayerDelta;
  method_ptr     = &NeuralNetwork::MomentumSGD;
  std::random_device rd;
  mt = std::mt19937( rd() );

  int gene_length = 0;
  for(int i = 0 ; i < (int)nLayerNeurons.size() - 1; i++){
    gene_length += nLayerNeurons[i].size() * nLayerNeurons[i+1].size()
      + nLayerNeurons[i+1].size() ;
  }
  
}


double NeuralNetwork::Sigmoid(double x){
  double y = 1. / ( 1 + std::exp( - x ) );
  if(y == 1){
    y = 0.9999999999999999;
  }else if(y == 0){
    y = 1e-323;
  }

  return y;
}

double NeuralNetwork::ReLU(double x){
  if(x >= 0){
    return x;
  }else{
    return 0;
  }
}

double NeuralNetwork::Identity(double x){
  return x;
}

double NeuralNetwork::DSigmoid(double x){
  return Sigmoid(x) * (1 - Sigmoid(x) );
}

double NeuralNetwork::DReLU(double x){
  if(x >=0){
    return 1;
  }else{
    return 0;
  }
}

double NeuralNetwork::DIdentity(double x){
  return 1;
}

void NeuralNetwork::Sigmoid(vec1D &lastLayer, vec1D const &beforeActF){
  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    lastLayer[i] = 1. / ( 1 + std::exp( - beforeActF[i] ) ) ;
    if(lastLayer[i] == 1){
      lastLayer[i] = 0.9999999999999999;
    }
    if(lastLayer[i] == 0){
      lastLayer[i] = 1e-323;
    }
  }
  return;
}


void NeuralNetwork::Softmax(vec1D &lastLayer, vec1D const &beforeActF){
  auto it_max = std::max_element(beforeActF.begin(), beforeActF.end());
  double denominator = 0;
  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    denominator += std::exp(beforeActF[i] - *it_max);
  }
  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    double numerator = std::exp(beforeActF[i] - *it_max);
    if(numerator == 0) numerator = 1e-323;
    lastLayer[i] = numerator / denominator;
  }
  return;
}


void NeuralNetwork::Identity(vec1D &lastLayer, vec1D const &beforeActF){

  for(int i = 0 ; i < (int)beforeActF.size() ; i++){
    lastLayer[i] = beforeActF[i];
  }
  return;
}

void NeuralNetwork::SetActivationFunction_Hidden(std::string actFuncHidden){
  if(actFuncHidden == "Sigmoid"){
    hidf_ptr  = &NeuralNetwork::Sigmoid;
    hiddf_ptr = &NeuralNetwork::DSigmoid;
    funcNameHidden = actFuncHidden;
  }else if(actFuncHidden == "ReLU"){
    hidf_ptr  = &NeuralNetwork::ReLU;
    hiddf_ptr = &NeuralNetwork::DReLU;
    funcNameHidden = actFuncHidden;
  }else if(actFuncHidden == "Identity"){
    hidf_ptr  = &NeuralNetwork::Identity;
    hiddf_ptr = &NeuralNetwork::DIdentity;
    funcNameHidden = actFuncHidden;
  }else{
    std::cout << "No valid function name was input !" << std::endl;
  }

}

void NeuralNetwork::SetActivationFunction_Output(std::string actFuncOutput){
  if(actFuncOutput == "Sigmoid"){
    outf_ptr  = &NeuralNetwork::Sigmoid;
    outdf_ptr = &NeuralNetwork::DSigmoid;
    deltaCalcu_ptr = &NeuralNetwork::CalculationLastLayerDelta;
    funcNameOutput = actFuncOutput;
  }else if(actFuncOutput == "Softmax"){
    outf_ptr  = &NeuralNetwork::Softmax;
    deltaCalcu_ptr = &NeuralNetwork::CalculationLastLayerDeltaSoftmax;
    outdf_ptr = NULL;
    funcNameOutput = actFuncOutput;
  }else if(actFuncOutput == "Identity"){
    outf_ptr  = &NeuralNetwork::Identity;
    outdf_ptr = &NeuralNetwork::DIdentity;
    deltaCalcu_ptr = &NeuralNetwork::CalculationLastLayerDelta;
    funcNameOutput = actFuncOutput;
  }else{
    std::cout << "No valid function name was input !" << std::endl;
  }

}

void NeuralNetwork::SetLossFunction(std::string nameLossFunction_){
  if(nameLossFunction_ == "MSE"){
    loss_ptr  = &NeuralNetwork::MeanSquaredError;
    dloss_ptr = &NeuralNetwork::DMSE;
  }else if(nameLossFunction_ == "BCE"){
    loss_ptr  = &NeuralNetwork::BinaryCrossEntropy;
    dloss_ptr = &NeuralNetwork::DBCE;
  }else if(nameLossFunction_ == "CCE"){
    loss_ptr  = &NeuralNetwork::CategoricalCrossEntropy;
    dloss_ptr = &NeuralNetwork::DCCE;
  }else{
    std::cout << "No valid function name was input !" << std::endl;
  }

}

void NeuralNetwork::SetLearningMethod(std::string method){
  if(method == "MomentumSGD"){
    method_ptr = &NeuralNetwork::MomentumSGD;
  }else if(method == "Adam"){
    method_ptr = &NeuralNetwork::Adam;
  }else if(method == "SGD"){
    method_ptr = &NeuralNetwork::SGD;
  }else{
    std::cout << "No valid method name was input !" << std::endl;
  }
   
}

void NeuralNetwork::CalcuHiddenLayer(){
  int nConnection = w.size();
  for(int ith_connection = 0 ; ith_connection < nConnection - 1 ; ith_connection++){
    for(int i = 0 ; i < (int)w[ith_connection].size() ; i++){
      double sum = 0;
      for(int j = 0 ; j < (int)w[ith_connection][i].size() ; j++){
	sum += nLayerNeurons[ith_connection][j] * w[ith_connection][i][j];
      }
      sum += b[ith_connection][i];
      beforeActFunc[ith_connection + 1][i] = sum;
      nLayerNeurons[ith_connection + 1][i] = hidf_ptr(sum);
    }
  }
  return;
}

void NeuralNetwork::CalculationAllStageOfLayer(){
  // --- Hidden Layer -- //
  CalcuHiddenLayer();
  // --- Last Layer --//
  int last_con = (int)w.size() - 1;

  for(int i = 0 ; i < (int)w[last_con].size() ; i++){ 
    double sum = 0;
    for(int j = 0 ; j < (int)w[last_con][i].size() ; j++){
      sum += nLayerNeurons[last_con][j] * w[last_con][i][j];
    }
    sum += b[last_con][i];
    beforeActFunc[last_con + 1][i] = sum;
  }
  outf_ptr(nLayerNeurons[last_con + 1], beforeActFunc[last_con + 1]);
  
}

template <class T>
void NeuralNetwork::InputData(std::vector<T> const &indata){
  if(nLayerNeurons[0].size() != indata.size()){
    std::cout << "-- Error ! The number of size is wrong !! --  " << std::endl;
    return;
  }
  for(int i = 0 ; i < (int)indata.size(); i++){
    nLayerNeurons[0][i] = indata[i];
  }
  return;
}

void NeuralNetwork::ParameterInitialization(){
  std::uniform_real_distribution<> rand_real(lower, upper);
  for(int i = 0 ; i < (int)w.size() ; i++){
    for(int j = 0 ; j < (int)w[i].size() ; j++){
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){
	w[i][j][k] = rand_real(mt);
      }
    }
  }

  for(int i = 0 ; i < (int)b.size() ; i++){
    for(int j = 0 ; j < (int)b[i].size() ; j++){
      b[i][j] = rand_real(mt);
    }
  }
  return;
}

void NeuralNetwork::TrainNN(vec2D const &inputDataSet, vec2D const &answerDataSet, int nRepetitions){
  long long nEntries = (long long)inputDataSet.size();
  if(nEntries != (int)answerDataSet.size()){
    std::cout << "-- Error !! Number of Entries are different between input data and answer data !! --" << std::endl;
    return;
  }
  int last = (int)nLayerNeurons.size() - 1;
  std::uniform_int_distribution<> uni(0, nEntries - 1);
  ParameterInitialization();
  double acc = 0;
  for(int iLearn = 0 ; iLearn < nRepetitions ; iLearn++){
    int iEntry = uni(mt);
    InputData(inputDataSet[iEntry]);	
    CalculationAllStageOfLayer();
    double error = loss_ptr( nLayerNeurons[last], answerDataSet[iEntry]) / numLastNeurons;
    acc += error;
    if(iLearn % printFreq == 0 && iLearn != 0 ){
      std::cout << "Loss (Average) = " << acc / printFreq << std::endl;
      losses.push_back(acc/printFreq);
      acc = 0;
    }
    
    //CalculationLastLayerDelta(nLayerNeurons[last], answerDataSet[iEntry]);

    (this ->*deltaCalcu_ptr)(nLayerNeurons[last], answerDataSet[iEntry]);
    CalculationAllLayerDelta();
    //-- here back probagation -- //
    CalcuGradientW();
    //BackPropagation();
    (this ->*method_ptr)();

  }


}

void NeuralNetwork::CalculationLastLayerDelta(vec1D const &NNoutput, vec1D const &answer){
  std::size_t last = delta.size() - 1 ;
  //std::vector<double> vDOutFunc(delta.size());
  //outdf_ptr(vDOutFunc, beforeActFunc[last]);
  for(int i = 0 ; i < (int)delta[last].size() ; i++){
    delta[last][i] = dloss_ptr(NNoutput[i], answer[i]) * outdf_ptr(beforeActFunc[last][i]);
  }
  return;
}

void NeuralNetwork::CalculationLastLayerDeltaSoftmax(vec1D const &NNoutput, vec1D const &answer){
  int last = delta.size() - 1;
  int K = nLayerNeurons[last].size();
  for(int i = 0 ; i < K ; i++){ // to determine delta[i]
    double acc = 0;
    for(int k = 0 ; k < K ; k++){
      if(i == k){
	acc += dloss_ptr(NNoutput[k], answer[k]) * NNoutput[i] * ( 1 - NNoutput[i] );
      }else{
	acc += dloss_ptr(NNoutput[k], answer[k]) * ( - NNoutput[i] * NNoutput[k]);
      }
    }
    delta[last][i] = acc;
  }
  return;
}

void NeuralNetwork::CalculationAllLayerDelta(){
  int last = delta.size() - 1;
  // delta[last] was already calculated in function "CalculationLastLayerDelta".
  for(int L = last - 1 ; L >= 1 ; L--){
    for(int i = 0 ; i < (int)delta[L].size() ; i++){
      double acc = 0;
      for(int k = 0 ; k < (int)delta[L + 1].size() ; k++){
	acc += delta[L + 1][k] * w[L][k][i];
      }
      delta[L][i] = acc * hiddf_ptr(beforeActFunc[L][i]); 
    }
  }
  return;
}

void NeuralNetwork::CalcuGradientW(){
  for(int L = 0 ; L < (int)w.size() ; L++){
    for(int i = 0 ; i < (int)w[L].size(); i++){
      for(int j = 0 ; j < (int)w[L][i].size() ; j++){
	dw[L][i][j] = delta[L + 1][i] * nLayerNeurons[L][j];
      }
      db[L][i] = delta[L + 1][i];
    }
  }
  return;
}

void NeuralNetwork::SGD(){
  for(int L = 0 ; L < (int)w.size() ; L++){
    for(int i = 0 ; i < (int)w[L].size(); i++){
      for(int j = 0 ; j < (int)w[L][i].size() ; j++){
	w[L][i][j] = w[L][i][j] - learningRate * dw[L][i][j];
      }
      b[L][i] = b[L][i] - learningRate * db[L][i];
    }
  }
  return;
}

void NeuralNetwork::MomentumSGD(){
  for(int L = 0 ; L < (int)w.size() ; L++){
    for(int i = 0 ; i < (int)w[L].size(); i++){
      for(int j = 0 ; j < (int)w[L][i].size() ; j++){
	double tempW = dw[L][i][j];
	dweight[L][i][j] = tempW * learningRate + alpha * dweight[L][i][j];
	w[L][i][j] = w[L][i][j] - dweight[L][i][j];
      }
      double tempB = db[L][i];
      doffset[L][i] = tempB * learningRate + alpha * doffset[L][i];
      b[L][i] = b[L][i] - doffset[L][i];
    }
  }
  return;
}

void NeuralNetwork::Adam(){
  static int t = 1;
  double hat_deno_beta_1 = (1 - std::pow(beta_1, t) );
  double hat_deno_beta_2 = (1 - std::pow(beta_2, t) );
  for(int L = 0 ; L < (int)w.size() ; L++){
    for(int i = 0 ; i < (int)w[L].size(); i++){
      for(int j = 0 ; j < (int)w[L][i].size() ; j++){
	m_w[L][i][j] = beta_1 * m_w[L][i][j] + (1 - beta_1) * dw[L][i][j];
	v_w[L][i][j] = beta_2 * v_w[L][i][j] + (1 - beta_2) * dw[L][i][j] * dw[L][i][j];
	double m_hat = m_w[L][i][j] / hat_deno_beta_1;
	double v_hat = v_w[L][i][j] / hat_deno_beta_2;
	w[L][i][j] = w[L][i][j] - alpha_Adam * m_hat / (std::sqrt(v_hat) + epsilon);
      }
      m_b[L][i] = beta_1 * m_b[L][i] + (1 - beta_1) * db[L][i];
      v_b[L][i] = beta_2 * v_b[L][i] + (1 - beta_2) * db[L][i] * db[L][i];
      double m_hat = m_b[L][i] / hat_deno_beta_1;
      double v_hat = v_b[L][i] / hat_deno_beta_2;
      b[L][i] = b[L][i] - alpha_Adam * m_hat / (std::sqrt(v_hat) + epsilon);
    }
  }
  t++;
  return;

}

void NeuralNetwork::LearningGA(vec2D const &inputDataSet, vec2D const &answerDataSet, int nRepetitions){
  long long nEntries = (long long)inputDataSet.size();
  if(nEntries != (int)answerDataSet.size()){
    std::cout << "-- Error !! Number of Entries are different between input data and answer data !! --" << std::endl;
    return;
  }
  
  int last = (int)nLayerNeurons.size() - 1;

  // -- From here, Learning by Genetic Algorithm -- //
  int gene_length = 0;
  for(int i = 0 ; i < (int)nLayerNeurons.size() - 1; i++){
    gene_length += nLayerNeurons[i].size() * nLayerNeurons[i+1].size()
      + nLayerNeurons[i+1].size() ;
  }
  std::cout << "gene length = " << gene_length << std::endl;
  GA = GeneticAlgorithm<double>(gene_length, population_);
  GA.GeneInitialization(lower, upper);

  for(int iLearn = 0 ; iLearn < nRepetitions ; iLearn++){
    if(iLearn % 100 == 0){
      std::cout << "------ " << iLearn << " Times Learning ----" << std::endl;
    }
    for(int iCreature = 0 ; iCreature < GA.GetPopulation() ;  iCreature++){
      SetWeightFromGene(GA, iCreature);
      double error = 0;
      for(int iEntry = 0 ; iEntry < nEntries ; iEntry++){
	InputData(inputDataSet[iEntry]);	
	CalculationAllStageOfLayer();
	error += loss_ptr( nLayerNeurons[last], answerDataSet[iEntry]) / numLastNeurons;
      }// end of Data Entry
#ifdef DBPR      
      if(iLearn % 100 == 0 && iCreature == 0)
	std::cout << "Error (Average) = " << error / nEntries << std::endl;
#endif
      GA.GiveScore(iCreature, error);
    }// End of looking into every creature
    //ShowInputAndOutput(GA, 0, inputDataSet);
    GA.CrossOver(nDominantGene, mutation_prob, "Minimize");
  }// End of Learning Repetition
  
  SetWeightFromGene(GA, 0);
}

void NeuralNetwork::LearningGA(vec2D const &inputDataSet, vec2D const &answerDataSet, double threshold){
  long long nEntries = (long long)inputDataSet.size();
  if(nEntries != (int)answerDataSet.size()){
    std::cout << "-- Error !! Number of Entries are different between input data and answer data !! --" << std::endl;
    return;
  }
  
  int last = (int)nLayerNeurons.size() - 1;

  // -- From here, Learning by Genetic Algorithm -- //
  int gene_length = 0;
  for(int i = 0 ; i < (int)nLayerNeurons.size() - 1; i++){
    gene_length += nLayerNeurons[i].size() * nLayerNeurons[i+1].size()
      + nLayerNeurons[i+1].size() ;
  }
  
  GA = GeneticAlgorithm<double>(gene_length, population_);
  GA.GeneInitialization(lower, upper);

  long long iLearn = 0;
  while(1){
    if(iLearn % 100 == 0){
      std::cout << "------ " << iLearn << " Times Learning ----" << std::endl;
    }
    for(int iCreature = 0 ; iCreature < GA.GetPopulation() ;  iCreature++){
      SetWeightFromGene(GA, iCreature);
      double error = 0;
      for(int iEntry = 0 ; iEntry < nEntries ; iEntry++){
  	InputData(inputDataSet[iEntry]);	
  	CalculationAllStageOfLayer();
  	error += loss_ptr( nLayerNeurons[last], answerDataSet[iEntry]) / numLastNeurons;
      }// end of Data Entry
#ifdef DBPR      
      if(iLearn % 100 == 0 && iCreature == 0) std::cout << "Error : " << error << std::endl;
#endif
      GA.GiveScore(iCreature, error);
    }// End of looking into every creature

    GA.CrossOver(nDominantGene, mutation_prob, "Minimize");
    if(GA.GetScore(0) < threshold) break;
    
    iLearn++;
  }// End of Learning Repetition

  SetWeightFromGene(GA, 0);
}

void NeuralNetwork::SetWeightFromGene(GeneticAlgorithm<double> &GA, int ith_creature){
  
  auto it = GA.GetGeneIterator(ith_creature);
  
  for(int i = 0 ; i < (int)w.size(); i++){ // i th connections
    for(int j = 0 ; j < (int)w[i].size() ; j++){ // j th neuron
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){ // k th node
	w[i][j][k] = *it;
	it++;
      }
      b[i][j] = *it;
      it++;
    }
  }
  return;
}

double NeuralNetwork::MeanSquaredError(vec1D const &lastNeurons, vec1D const &answerData){
  if(lastNeurons.size() != answerData.size()){
    std::cout << "-- Error !! Discrepancy between number of neurons in last layer and answer data -- " << std::endl;
    return -1;
  }
  double acc = 0;
  auto Square = [](double x){return x * x;};
  for(int i = 0 ; i < (int)lastNeurons.size() ; i++){
    acc += Square(lastNeurons[i] - answerData[i]);
  }
  return acc;
}


double NeuralNetwork::BinaryCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData){
  if((int)lastNeurons.size() != 1){
    std::cout << "Error ! The number of neurons in Last Layer should be 1 if you use Binary Cross Entropy for loss function." << std::endl;
  }
  if(lastNeurons[0] == 1){
    double bce = - answerData[0] * std::log(0.9999999999999999)
      - (1 - answerData[0]) * std::log(1 -  0.9999999999999999);
    return bce;
  }
  double BCE = - answerData[0] * std::log(lastNeurons[0])
    - (1 - answerData[0]) * std::log(1 - lastNeurons[0]);
  return BCE; 
}

double NeuralNetwork::CategoricalCrossEntropy(vec1D const &lastNeurons, vec1D const &answerData){
  if(lastNeurons.size() != answerData.size()){
    std::cout << "-- Error !! Discrepancy between number of neurons in last layer and answer data -- " << std::endl;
    return -1;
  }
  double acc = 0;
  for(int i = 0 ; i < (int)answerData.size() ; i++){
    if(lastNeurons[i] == 1){
      double bce = - answerData[i] * std::log(0.9999999999999999)
	- (1 - answerData[i]) * std::log(1 -  0.9999999999999999);
      acc += bce;

      continue;
    }
    acc +=
      - answerData[i] * std::log(lastNeurons[i])
      - (1 - answerData[i]) * std::log(1. - lastNeurons[i]) ;

  }
  return acc;
}

double NeuralNetwork::DMSE(double y, double d){
  return 2 * (y - d);
}

double NeuralNetwork::DBCE(double y, double d){
  if(y == 1) y = 0.9999999999999999;
  return -1 * ( d - y ) / ( y * ( 1 - y ) );
}

double NeuralNetwork::DCCE(double y, double d){
  if(y == 1) y = 0.9999999999999999;
  return -1 * ( d - y ) / ( y * ( 1 - y ) );
}

void NeuralNetwork::ShowInputAndOutput(GeneticAlgorithm<double> &GA, int ith_creature, vec2D const &input){
  SetWeightFromGene(GA, ith_creature);
  for(int i = 0 ; i < (int)input.size() ; i++){
    std::cout << "Data ( ";
    for(auto it = input[i].begin() ; it != input[i].end(); it++){
      if(it + 1 != input[i].end()){
	std::cout << *it  << ", ";
      }else if(it + 1 == input[i].end()){
	std::cout << *it << " ";
      }
    }
    std::cout << ") --> ";
    std::cout << " Output : (" ;
    int last = (int)nLayerNeurons.size() - 1;
    InputData(input[i]);
    CalculationAllStageOfLayer();
    for(auto it = nLayerNeurons[last].begin() ;it != nLayerNeurons[last].end() ;it++){
      if(it != nLayerNeurons[last].end() - 1){
	std::cout << *it << ", ";
      }else{
	std::cout << *it << " ";
      }
    }
    std::cout << ")" << std::endl;
  }

}


void NeuralNetwork::PrintWeightMatrix(){
  for(int i = 0 ; i < (int)w.size() ; i++){
    std::cout << "-- Layer " << i << " to " << i+1 << " --" << std::endl;
    for(int j = 0 ; j < (int)w[i].size() ; j++){
      std::cout << "(" ;
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){
	std::cout << std::fixed << std::setprecision(6);
	std::cout << std::right << std::setw(10);
	std::cout << w[i][j][k] << " ";
      }
      std::cout << ")";

      std::cout << " ( " << std::right << std::setw(10)  << b[i][j] << " ) " << std::endl;
      
    }
  }
  return;
}

void NeuralNetwork::PrintLastLayer(vec2D const &inputDataSet){
  for(int i = 0 ; i < (int)inputDataSet.size() ; i++){
    std::cout << "Data ( ";
    for(auto it = inputDataSet[i].begin() ; it != inputDataSet[i].end(); it++){
      if(it + 1 != inputDataSet[i].end()){
	std::cout << *it  << ", ";
      }else if(it + 1 == inputDataSet[i].end()){
	std::cout << *it << " ";
      }
    }
    std::cout << ") --> ";
    std::cout << " Output : ( " ;
    int last = (int)nLayerNeurons.size() - 1;
    InputData(inputDataSet[i]);
    CalculationAllStageOfLayer();
    for(auto it = nLayerNeurons[last].begin() ;it != nLayerNeurons[last].end() ;it++){
      if(it != nLayerNeurons[last].end() - 1){
	std::cout << *it << ", ";
      }else{
	std::cout << *it << " ";
      }
    }
    std::cout << ")" << std::endl;
  }  

}

void NeuralNetwork::PrintLastLayer(vec1D const &inputData){
  std::cout << "Data ( ";
  for(auto it = inputData.begin() ; it != inputData.end(); it++){
    if(it + 1 != inputData.end()){
      std::cout << *it  << ", ";
    }else if(it + 1 == inputData.end()){
      std::cout << *it << " ";
    }
  }
  std::cout << ") --> ";
  std::cout << " Output : ( " ;
  int last = (int)nLayerNeurons.size() - 1;
  InputData(inputData);
  CalculationAllStageOfLayer();
  for(auto it = nLayerNeurons[last].begin() ;it != nLayerNeurons[last].end() ;it++){
    if(it != nLayerNeurons[last].end() - 1){
      std::cout << *it << ", ";
    }else{
      std::cout << *it << " ";
    }
  }
  std::cout << ")" << std::endl;

}

vec1D::iterator NeuralNetwork::GetOutputIterator(vec1D const &inputData){
  InputData(inputData);
  CalculationAllStageOfLayer();
  std::size_t size = nLayerNeurons.size();
  return nLayerNeurons[size - 1].begin();
}


void NeuralNetwork::SaveNeuralNetwork(std::string output_filename){
  std::ofstream ofile;
  std::string fname_with_extension = output_filename + ".txt";
  ofile.open(fname_with_extension, std::ios::out);
  ofile << structure << std::endl;
  ofile << funcNameHidden << std::endl;
  ofile << funcNameOutput << std::endl;
  ofile << (*(losses.end() - 1)) << std::endl;
  for(int i = 0 ; i < (int)w.size() ; i++){
    for(int j = 0 ; j < (int)w[i].size() ; j++){
      for(int k = 0 ; k < (int)w[i][j].size() ; k++){
	ofile << i << " " << j << " " << k << " ";
	ofile << std::scientific << std::setprecision(15) << w[i][j][k] << std::endl;
      }
    }
  }
  ofile << "b" << std::endl;
  for(int i = 0 ; i < (int)b.size() ; i++){
    for(int j = 0 ; j < (int)b[i].size() ; j++){
      ofile << i << " " << j << " ";
      ofile << std::scientific << b[i][j] << std::endl;
    }
  }
  ofile << "end" << std::endl;
  std::cout << "Created Parameter file \"" << output_filename << ".txt\"" << std::endl;
  return;
}

void NeuralNetwork::ReadWeightMatrix(std::string filename){
  std::ifstream ifile(filename, ios::in);
  if(!ifile){
    std::cout << "Error in reading file \"" << filename << "\"" << std::endl;
  }
  std::string buffer;
  std::getline(ifile, buffer);
  if(buffer != structure){
    std::cout << "Error ! The structure of neural network is wrong! " << std::endl;
    return;
  }
  std::getline(ifile, buffer);
  SetActivationFunction_Hidden(buffer);
  std::getline(ifile, buffer);
  SetActivationFunction_Output(buffer);
  std::getline(ifile, buffer);
  while( std::getline(ifile, buffer) ){
    if(buffer == "b"){
      break;
    }
    std::istringstream iss(buffer.data());
    int i, j, k;
    double weight;
    iss >> i >> j >> k >> weight;
    w[i][j][k] = weight;
  }
  while(std::getline(ifile, buffer)){
    if(buffer == "end"){
      break;
    }
    std::istringstream iss(buffer.data());
    int i, j;
    double offset;
    iss >> i >> j >> offset;
    b[i][j] = offset;
  }
  
  std::cout << "Reading weight and bias parameters has been done successfully." << std::endl;
}

