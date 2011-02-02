package edu.berkeley.nlp.assignments.assign1.student;

import java.util.List;

import edu.berkeley.nlp.langmodel.LanguageModelFactory;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;

public class ExactLmFactory implements LanguageModelFactory {

   public NgramLanguageModel newLanguageModel(
         Iterable<List<String>> trainingData) {
      System.out.println("Exact language model");
      KneserNeyTrigramLm model = new KneserNeyTrigramLm(trainingData, false);
      return model;
   }

}
