package edu.berkeley.nlp.assignments.assign1.student;

/**
 * A very compact hash map. It is so compact that it has only one array for
 * data. Both the keys and the values are packed in the same data array.
 * 
 * @author rxin
 * 
 */
public class CrazilyPackedHashMap {
   
   // 0x3FFFFF = 22 bits of 1.
   protected long valueMask = 0x3FFFFF;
   protected long keyMask = ~valueMask;
   protected int numFirstKeyBits = 23;
   protected int numSecondKeyBits = 19;
   protected int numValueBits = 64 - numFirstKeyBits - numSecondKeyBits;
   
   int size;
   long data[];

   /**
    * Initializes the hash table to a prime capacity which is at least
    * <tt>initialCapacity/loadFactor + 1</tt>.
    */
   public CrazilyPackedHashMap(int initialCapacity, float loadFactor,
         long valueMask, int numFirstKeyBits, int numSecondKeyBits) {
      
      // Initialize the data array.
      int length = HashFunctions.fastCeil(initialCapacity / loadFactor);
      length = PrimeFinder.nextPrime(length);
      System.out.println("TrigramCounter length: " + length);
      data = new long[length];
      
      // Set the parameters.
      this.valueMask = valueMask;
      keyMask = ~valueMask;
      this.numFirstKeyBits = numFirstKeyBits;
      this.numSecondKeyBits = numSecondKeyBits;
      this.numValueBits = 64 - numFirstKeyBits - numSecondKeyBits;
   }

   /**
    * Return the number of unique trigrams.
    */
   public int size() {
      return size;
   }
   
   /**
    * Return the value of (key1, key2).
    */
   public int get(int key1, int key2) {
      long key = getKey(key1, key2);
      int location = locatePosition(key);
      long value = data[location] & valueMask;
      return (int) value;
   }
   
   /**
    * Adjust or put the value and return the value.
    */
   public int adjustOrPutValue(int key1, int key2, int adjustAmount) {
      long key = getKey(key1, key2);
      int location = locatePosition(key);
      long value = data[location] & valueMask;
      
      if (value == 0) {
         size++;
      }
      value += adjustAmount;
      data[location] = key | value;
      return (int)value;
   }
   
   /**
    * Return the sum of all values.
    */
   public long sum() {
      long sum = 0;
      for (int i = 0; i < data.length; i++) {
         long value = data[i] & valueMask;
         sum += value;
      }
      return sum;
   }
   
   protected long getKey(int key1, int key2) {
      long key = key1;
      key = key << numSecondKeyBits | key2;
      key = key << numValueBits;
      return key;
   }
   
   protected int locatePosition(long key) {
      int index = HashFunctions.hash(key) % data.length;
      //int index = (int)(key % (long)data.length);
      if (index < 0) index = -index;
      while ((data[index] != 0) && (data[index] & keyMask) != key) {
         index++;
         if (index == data.length) index = 0;
      }
      return index;
   }

}
