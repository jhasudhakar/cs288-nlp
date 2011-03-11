package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;

public class HeuristicAlignerFactory implements WordAlignerFactory {

   public WordAligner newAligner(Iterable<SentencePair> trainingData) {
      HeuristicAligner aligner = new HeuristicAligner();
      aligner.train(trainingData);
      return aligner;
   }

}
