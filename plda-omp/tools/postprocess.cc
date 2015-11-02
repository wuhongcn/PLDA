/*
  An example running of this program:
  ./postpreprocess --vocab_file url.dat.plda_vocab        \
                   --input_model url.dat.plda_model_index \
                   --output_model url.dat.plda_model
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

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::map; 
using std::string;
using std::vector;
using std::sort;


int main(int argc, char** argv) 
{
  string vocab_file;
  string output_file;
  string model_file;
  if (argc != 7) {
    std::cout << "Usage: postprocess --vocab_file vocab_file --input_model index_model_file --output_model model_file" << std::endl;
    return -1;
  }
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "--vocab_file")) {
      vocab_file = argv[i+1];
      ++i;
    }else if(0 == strcmp(argv[i], "--input_model")) {
      model_file = argv[i+1];
      ++i;
    }else if(0 == strcmp(argv[i], "--output_model")) {
      output_file = argv[i+1];
      ++i;
    }else{
      std::cout << "Usage: postprocess --vocab_file vocab_file --input_model index_model_file --output_model model_file" << std::endl;
      return -1;
    }
  }

  map<long, string> word_index_map;
  ifstream fin(vocab_file.c_str());
  string line;
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      istringstream ss(line);
      string word;
      int word_index;
      while (ss >> word >> word_index) {  // Load and init a document.
        word_index_map[word_index] = word;
      }
    }
  }

  std::cout << "phase 2" << std::endl;
 
  ifstream min(model_file.c_str());
  std::ofstream fout(output_file.c_str());
  while (getline(min, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      istringstream ss(line);
      string word;
      int word_index;
      int count;
      ss >> word_index;
      fout << word_index_map[word_index] << "\t";
      while (ss >> count) {  // Load and init a document.
        fout << count << " ";
      }
      fout << std::endl;
    }
  }

  return 0;
}
