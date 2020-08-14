#include <iostream>
#include <cmath>
#include <vector>
#include <iterator>
#include <random>

#ifndef _GeneticAlgorithm_
#define _GeneticAlgorithm_

// You should use <int> or <double>.

template <typename T>
class GeneticAlgorithm{
public:
  GeneticAlgorithm(){};
  GeneticAlgorithm(int gene_length, int population);
  ~GeneticAlgorithm(){};

  
  void GiveScore(int ith_creature, double score);
  double GetScore(int ith_creature){
    return creatures[ith_creature].Score;
  };

  void GeneInitialization(int min, int max);
  void GeneInitialization(double min, double max);

  int GetGeneLength(){return gene_length_;};
  int GetPopulation(){return population_;};
  
  typename std::vector<T>::iterator GetGeneIterator(int ith_creature);

  void CrossOver(int numDominantGene, double mutation_prob, std::string optimization_option);
  
  private:
  
  int gene_length_;
  int population_;

  T min_;
  T max_;
  
  std::mt19937 mt;
  T type_keeper;
  struct Creature{
    std::vector<T> gene;
    double Score;
  };

  std::vector<Creature> creatures;  

  void Mutation(std::vector<int> &v);
  void Mutation(std::vector<double> &v);

};

#endif
