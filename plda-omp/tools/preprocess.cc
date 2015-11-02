/*
  An example running of this program:
  preprocess --input url.dat.plda            \
             --output url.dat.plda_index     \
             --vocab_file url.dat.plda_vocab \
             --threshold 10
*/

#include <cstdlib>
#include <fstream>
#include <map>

#include <vector>

#include <string>
#include <string.h>

#include <algorithm>

#include <iostream>
#include <sstream>

#include <sys/time.h>

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::map; 
using std::string;
using std::vector;

void sortMapByValue(std::map<std::string, long>& tMap, std::vector<std::pair<std::string, long> >& tVector);

int cmp(const std::pair<std::string, long>& x, const std::pair<std::string, long>& y);

int main(int argc, char** argv) 
{
  string corpus_file;
  string output_file;
  string vocab_file;
  struct timeval start1, end1, start, end;

  int threshold = 0;
  long unique_word_size = 0;

  gettimeofday(&start, NULL);
  if (argc < 7) {
    std::cout << "Usage: ./preprocess --input training_data_file --output index_data_file --vocab_file vocab_file --threshold 10" << std::endl;
    return -1;
  }
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "--input")) {
      corpus_file = argv[i+1];
      ++i;
    }else if(0 == strcmp(argv[i], "--output")) {
      output_file = argv[i+1];
      ++i;
    }else if(0 == strcmp(argv[i], "--vocab_file")) {
      vocab_file = argv[i+1];
      ++i;
    }else if(0 == strcmp(argv[i], "--threshold")) {
      std::istringstream(argv[i+1]) >> threshold;
      ++i;
    }else {
      std::cout << "Usage: ./preprocess --input training_data_file --output index_data_file --vocab_file vocab_file --threshold 10" << std::endl;
      return -1;
    }
  }

  map<string, long> word_stat_map;
  map<string, long> word_index_map;

  gettimeofday(&start1, NULL);
  ifstream fin(corpus_file.c_str());
  string line;
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      istringstream ss(line);
      string word;
      long count;
      while (ss >> word >> count) {  // Load and init a document.
        long word_index;
        map<string, long>::const_iterator iter = word_stat_map.find(word);
        if (iter == word_stat_map.end()) {
          word_stat_map[word] = count;
        } else {
          word_stat_map[word] += count;
        }
      }
    }
  }
  fin.close();

  gettimeofday(&end1, NULL);
  std::cout << "step1: get word_stat_map " << end1.tv_sec - start1.tv_sec + (end1.tv_usec - start1.tv_usec)/1000000.0 << " second." << std::endl;

  gettimeofday(&start1, NULL);
  vector<std::pair<string,long> > word_stat_vector;
  sortMapByValue(word_stat_map,word_stat_vector);

  std::ofstream sout(vocab_file.c_str());
  for(long i=0;i<word_stat_vector.size();i++) {
    if(word_stat_vector[i].second >= threshold) {
      sout << word_stat_vector[i].first.c_str() << " " << unique_word_size << "\n";
      word_index_map[word_stat_vector[i].first] = unique_word_size++;
    }
  }
  sout.close();
  gettimeofday(&end1, NULL);
  std::cout << "step2: get word_index " << end1.tv_sec - start1.tv_sec + (end1.tv_usec - start1.tv_usec)/1000000.0 << " second." << std::endl;

  gettimeofday(&start1, NULL);
  fin.open(corpus_file.c_str());
  std::ofstream fout(output_file.c_str());
  fout << "# " << unique_word_size << "\n";
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      istringstream ss(line);
      string word;
      long count;
      while (ss >> word >> count) {  // Load and init a document.
        long word_index, word_count;
        word_count = word_stat_map[word];
        if (word_count >= threshold) {
          fout << word_index_map[word] << " " << count << " ";
        }
      }
      fout << "\n";
    }
  }
  fout.close();
  fin.close();
  gettimeofday(&end1, NULL);
  std::cout << "step3: get new training file " << end1.tv_sec - start1.tv_sec + (end1.tv_usec - start1.tv_usec)/1000000.0 << " second." << std::endl;

  std::cout << "Unique word size: " << unique_word_size << std::endl;
  gettimeofday(&end,NULL);
  std::cout << "Total elapsed time: " << end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec)/1000000.0 << " second." << std::endl;

  return 0;
}

 

int cmp(const std::pair<std::string, long>& x, const std::pair<std::string, long>& y)
{
  return x.second > y.second;
}

 

void sortMapByValue(std::map<std::string, long>& tMap, std::vector<std::pair<std::string, long> >& tVector)
{

  for (std::map<std::string, long>::iterator curr = tMap.begin(); curr != tMap.end(); curr++) {
     tVector.push_back(std::make_pair(curr->first, curr->second));
  }

  std::sort(tVector.begin(), tVector.end(), cmp);
}

