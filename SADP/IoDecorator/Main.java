import java.io.*;
import java.io.IOException;

public class Main {

    public static void main(String[] args) {
        try {
            InputStream inputStream = new LowerCaseInputStream(new FileInputStream("example.txt"));
            int value;
            while ((value = inputStream.read()) != -1) {
                System.out.print((char) value);
            }
            inputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}