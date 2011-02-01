package edu.berkeley.nlp.assignments.assign1.student;

import java.util.List;

import edu.berkeley.nlp.langmodel.LanguageModelFactory;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;

public class ExactLmFactory implements LanguageModelFactory {

   public NgramLanguageModel newLanguageModel(
         Iterable<List<String>> trainingData) {
      ExactLm model = new ExactLm(trainingData);
      return model;
   }

}
