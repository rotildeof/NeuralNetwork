#include "GeneticAlgorithm.h"
#include <typeinfo>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <string>

//#define DEBUG
#ifndef _GASOURCE_
#define _GASOURCE_

template <typename T>
GeneticAlgorithm<T>::GeneticAlgorithm(int gene_length, int population)
  : gene_length_(gene_length), population_(population),
    creatures(population)
{
  for(int i = 0 ; i < (int)creatures.size() ; i++){
    creatures[i].gene.assign(gene_length, 0);
  }

  std::random_device rd;
  mt = std::mt19937(rd());
}

template <typename T>
void GeneticAlgorithm<T>::GiveScore(int ith_creature, double score){
  creatures[ith_creature].Score = score;
  return;
}

template <typename T>
typename std::vector<T>::iterator GeneticAlgorithm<T>::GetGeneIterator(int ith_creature){
  return creatures[ith_creature].gene.begin();
}

template <typename T>
void GeneticAlgorithm<T>::GeneInitialization(int min, int max){
  //#ifdef DEBUG
  std::cout << "Initialization <int> " << std::endl;
  //#endif
  assert(min < max);
  min_ = min;
  max_ = max;
  std::uniform_int_distribution<> uni(min, max);
  for(int i = 0 ; i < (int)creatures.size() ; i++){
    for(int j = 0 ; j < (int)creatures[i].gene.size() ; j++){
      creatures[i].gene[j] = uni(mt);
    }
  }
  return;
}

template <typename T>
void GeneticAlgorithm<T>::GeneInitialization(double min, double max){
  //#ifdef DEBUG
  std::cout << "Initialization <double> " << std::endl;
  //#endif
  assert(min < max);
  min_ = min;
  max_ = max;
  std::uniform_real_distribution<> uni(min, max);
  for(int i = 0 ; i < (int)creatures.size() ; i++){
    for(int j = 0 ; j < (int)creatures[i].gene.size() ; j++){
      creatures[i].gene[j] = uni(mt);
    }
  }
  return;
}

template <typename T>
void GeneticAlgorithm<T>::CrossOver(int numDominantGene,
				    double mutation_prob,
				    std::string optimization_option){
  assert(numDominantGene < creatures.size() && numDominantGene > 1);
  if(optimization_option == "Minimize"){
    std::sort(creatures.begin(), creatures.end(),
	      [](const Creature &x, const Creature &y){return x.Score < y.Score;});
  }else if(optimization_option == "Maximize"){
    std::sort(creatures.rbegin(), creatures.rend(),
	      [](const Creature &x, const Creature &y){return x.Score < y.Score;});
  }else{
    std::cout << "No valid optimization option was input" << std::endl;
    return;
  }

  std::vector<int> numbers(numDominantGene);
  std::iota(numbers.begin(), numbers.end(), 0);
  for(int i = numDominantGene ; i < (int)creatures.size(); i++){
    std::shuffle(numbers.begin(), numbers.end(), mt);
    for(int j = 0 ; j < (int)creatures[i].gene.size() ; j++){
      std::uniform_int_distribution<> uni(0, 1);
      int parent = uni(mt);
      creatures[i].gene[j] = creatures[parent].gene[j];
      
#ifdef DEBUG
      std::cout << parent;
#endif
    }
#ifdef DEBUG
    std::cout << "  i : " << i << std::endl;
#endif
    std::uniform_real_distribution<> uni_real(0, 1.0);
    // -- mutation judgement -- //
    double val = uni_real(mt);
    if(val < mutation_prob) Mutation(creatures[i].gene);
  }

}

template <typename T>
void GeneticAlgorithm<T>::Mutation(std::vector<int> &v){
  // swap two components chosen randomly
  //#ifdef DEBUG
  std::cout << "Mutation occurs <int> !!" << std::endl;
  //#endif
  std::size_t size = v.size();
  std::uniform_int_distribution<> uni(0, size - 1);
  int i1 = uni(mt);
  int i2 = uni(mt);

  while(i1 == i2){
    i2 = uni(mt);
  }
  std::iter_swap(v.begin() + i1, v.begin() + i2);
  
#ifdef DEBUG
  std::cout << "i1 --> " << i1 << std::endl;
  std::cout << "i2 --> " << i2 << std::endl;
#endif  
  
  // rewrite the value of a component

  std::uniform_int_distribution<> uni_val(min_, max_);
  int ii = uni(mt);
  int val = uni_val(mt);
  v[ii] = val;
#ifdef DEBUG
  std::cout << "rewrote : ii --> " << ii << " val --> " << val << std::endl;
#endif
  return;
}

template <typename T>
void GeneticAlgorithm<T>::Mutation(std::vector<double> &v){
  // swap two components chosen randomly
#ifdef DEBUG
  std::cout << "Mutation occurs <double> !!" << std::endl;
#endif
  std::size_t size = v.size();
  std::uniform_int_distribution<> uni(0, size - 1);
  int i1 = uni(mt);
  int i2 = uni(mt);

  while(i1 == i2){
    i2 = uni(mt);
  }
  std::iter_swap(v.begin() + i1, v.begin() + i2);
  
#ifdef DEBUG
  std::cout << "i1 --> " << i1 << std::endl;
  std::cout << "i2 --> " << i2 << std::endl;
#endif  
  
  // rewrite the value of a component

  std::uniform_real_distribution<> uni_val(min_, max_);
  
  int ii = uni(mt);
  double val = uni_val(mt);
  v[ii] = val;
  
#ifdef DEBUG
  std::cout << "rewrote : ii --> " << ii << " val --> " << val << std::endl;
#endif
  return;
}

#endif
