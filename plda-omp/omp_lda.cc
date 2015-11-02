// Copyright 2008 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/*
  An example running of this program:
  mpiexec -n 4 ./mpi_lda --num_topics 300                              \
                         --alpha 0.1                                   \
                         --beta 0.01                                   \ 
                         --training_data_file new_training_file_index  \
                         --model_file lda_model_index.txt              \
                         --total_iterations 15                         \
*/

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <string>

#include "common.h"
#include "document.h"
#include "model.h"
#include "accumulative_model.h"
#include "sampler.h"
#include "cmd_flags.h"

using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::set;
using std::vector;
using std::list;
using std::map;
using std::sort;
using std::string;
using learning_lda::LDADocument;

namespace learning_lda {

LDACmdLineFlags flags;
// A wrapper of MPI_Allreduce. If the vector is over 32M, we allreduce part
// after part. This will save temporary memory needed.
void AllReduceTopicDistribution(int myid, int64* buf, int count) {
  static int kMaxDataCount = 1 << 22;
  static int datatype_size = sizeof(*buf);
  if (count > kMaxDataCount) {
    char* tmp_buf = new char[datatype_size * kMaxDataCount];
   
    int i = 0;
    for (i = 0; i < count / kMaxDataCount; ++i) {
      int64 tmp_pos = datatype_size * kMaxDataCount * i;  
      MPI_Allreduce(reinterpret_cast<char*>(buf) +
             tmp_pos,
             tmp_buf,
             kMaxDataCount, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
      memcpy(reinterpret_cast<char*>(buf) +
             tmp_pos,
             tmp_buf,
             kMaxDataCount * datatype_size);
    }
    // If count is not divisible by kMaxDataCount, there are some elements left
    // to be reduced.
    if (count % kMaxDataCount > 0) {
      int64 tmp_pos = datatype_size * kMaxDataCount * i;
      MPI_Allreduce(reinterpret_cast<char*>(buf) +
               tmp_pos,
               tmp_buf,
               count - kMaxDataCount * (count / kMaxDataCount), 
               MPI_LONG_LONG, MPI_SUM,
               MPI_COMM_WORLD);
      memcpy(reinterpret_cast<char*>(buf) +
               tmp_pos,
               tmp_buf,
               (count - kMaxDataCount * (count / kMaxDataCount)) * datatype_size);
    }
    delete[] tmp_buf;
  } else {
    char* tmp_buf = new char[datatype_size * count];
    MPI_Allreduce(buf, tmp_buf, count, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    memcpy(buf, tmp_buf, datatype_size * count);
    delete[] tmp_buf;
  }
}

class ParallelLDAModel : public LDAModel {
 public:
  ParallelLDAModel(int num_topic, int unique_word_size)
      : LDAModel(num_topic, unique_word_size) {
  }
  void ComputeAndAllReduce(int myid, const LDACorpus& corpus) {
    for (list<LDADocument*>::const_iterator iter = corpus.begin();
         iter != corpus.end();
         ++iter) {
      LDADocument* document = *iter;
      for (LDADocument::WordOccurrenceIterator iter2(document);
           !iter2.Done(); iter2.Next()) {
        IncrementTopic(iter2.Word(), iter2.Topic(), 1);
      }
    }
    AllReduceTopicDistribution(myid, &memory_alloc_[0], memory_alloc_.size());
  }
};

int DistributelyLoadAndInitTrainingCorpus(
    LDACmdLineFlags* flags,
    int myid, int pnum, LDACorpus* corpus) {

  corpus->clear();
  ifstream fin(flags->training_data_file_.c_str());
  string line;
  int index = 0;
  while (getline(fin, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.

      if(index % 1000000 == 0)
          std::cout << "id= " << myid << ": " << index << ", " << corpus->size() << std::endl;

      if (index % pnum == myid) {
        // This is a document that I need to store in local memory.
        DocumentWordTopicsPB document;
        string word;
        int word_index;
        int count;
        istringstream ss(line);
        int document_size = 0;
        while (ss >> word_index >> count) { // Load and init a document.
          vector<int32> topics;
          for (int i = 0; i < count; ++i) {
            topics.push_back(RandInt(flags->num_topics_));
          }
          document.add_wordtopics(word, word_index, topics);
          document_size++;
        }
        if (document_size > 0) {
          corpus->push_back(new LDADocument(document, flags->num_topics_));
        }
      } else {
      
      }
      index++;
    } else if (line.size() > 0 &&
               line[0] == '#') {
      istringstream ss(line);
      string word;
      ss >> word >> flags->unique_word_size_;
    }
  }
  return corpus->size();
}

void FreeCorpus(LDACorpus* corpus) {
  for (list<LDADocument*>::iterator iter = corpus->begin();
       iter != corpus->end();
       ++iter) {
    if (*iter != NULL) {
      delete *iter;
      *iter = NULL;
    }
  }
}
}
int main(int argc, char** argv) {
  using learning_lda::LDACorpus;
  using learning_lda::LDAModel;
  using learning_lda::ParallelLDAModel;
  using learning_lda::LDASampler;
  using learning_lda::DistributelyLoadAndInitTrainingCorpus;
  using learning_lda::LDACmdLineFlags;
  int myid, pnum;
  double start, end, elapsed;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum);

  LDACmdLineFlags flags;
  flags.ParseCmdFlags(argc, argv);
  if (!flags.CheckParallelTrainingValidity()) {
    return -1;
  }

  srand(time(NULL));

  LDACorpus corpus;
  int unique_word_size;
  start = MPI_Wtime();
  CHECK_GT(DistributelyLoadAndInitTrainingCorpus(&flags,
                                     myid, pnum, &corpus), 0);
  unique_word_size = flags.unique_word_size_;
  end = MPI_Wtime();
  elapsed = end - start;
  std::cout << "id = " << myid << " init. time is " << elapsed << " sec." << std::endl;
                                   //  myid, pnum, &corpus, &allwords), 0);
  std::cout << "Training data loaded" << std::endl;
  // Make vocabulary words sorted and give each word an int index.
  
  ParallelLDAModel model(flags.num_topics_, unique_word_size);
  model.ComputeAndAllReduce(myid, corpus);
  LDASampler sampler(flags.alpha_, flags.beta_, &model, NULL);
  
  std::cout << "\n" << "the Number of Iteration is " <<  flags.total_iterations_ << " " << unique_word_size << " " << flags.num_topics_ <<" " <<unique_word_size << " "<<" \n";
#pragma omp parallel
{
  for (int iter = 0; iter < flags.total_iterations_; ++iter) {
#pragma omp master
{
    start = MPI_Wtime();
    if (myid == 0) {
      std::cout << "Iteration " << iter << " ...\n";
    }
}
    int tid = omp_get_thread_num();
    int tnum = omp_get_num_threads();
    sampler.DoIteration(myid, pnum, tid, tnum, &corpus, true, false);
#pragma omp barrier
#pragma omp master
{
    end = MPI_Wtime();
    elapsed = end - start;

    std::cout << "id = " << myid << "tnum = " << tnum << " iter = " << iter << " iteration is " << elapsed << " sec." << std::endl;
   
    start = MPI_Wtime();
    model.memory_alloc_.assign(((int64)flags.num_topics_)*((int64)unique_word_size + 1), 0);
    model.ComputeAndAllReduce(myid, corpus);
    end = MPI_Wtime();
    elapsed = end - start;

    std::cout << "id = " << myid << " iter = " << iter << " Compute and reduce time is " << elapsed << " sec." << std::endl;
}
#pragma omp barrier
  }
}
  if (myid == 0) {
    std::ofstream fout(flags.model_file_.c_str());
    model.AppendAsString(fout);
  }
  FreeCorpus(&corpus);
  MPI_Finalize();
  return 0;
}
