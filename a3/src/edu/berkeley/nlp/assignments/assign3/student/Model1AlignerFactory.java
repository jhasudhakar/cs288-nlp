package edu.berkeley.nlp.assignments.assign3.student;

import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.mt.WordAlignerFactory;

public class Model1AlignerFactory implements WordAlignerFactory
{

	public WordAligner newAligner(Iterable<SentencePair> trainingData) {
	   Model1SoftEmWeirdAligner aligner = new Model1SoftEmWeirdAligner();
	   aligner.train(trainingData);
	   return aligner;
	}

}
