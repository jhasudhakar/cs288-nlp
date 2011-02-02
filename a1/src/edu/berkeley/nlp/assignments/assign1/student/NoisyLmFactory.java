package edu.berkeley.nlp.assignments.assign1.student;

import java.util.List;

import edu.berkeley.nlp.langmodel.LanguageModelFactory;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;

public class NoisyLmFactory implements LanguageModelFactory
{

	public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData) {
	   System.out.println("Noisy language model");
	   
	   // Don't use approximation given our exact model fits all requirements.
      KneserNeyTrigramLm model = new KneserNeyTrigramLm(trainingData, false);
      return model;
	}

}
