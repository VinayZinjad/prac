import java.io.*;
import java.util.*;

class LowerCaseInputStream extends FilterInputStream {
    public LowerCaseInputStream(InputStream in) {
        super(in);
    }

    public int read() throws IOException {
        int c = super.read();
        return (c == -1 ? c : Character.toLowerCase((char) c));
    }

    public int read(byte[] b, int offset, int len) throws IOException {
        int result = super.read(b, offset, len);
        for (int i = offset; i < offset + result; i++) {
            b[i] = (byte) Character.toLowerCase();
        }
        return result;
    }
}

public class Main {
    public static void main(String[] args) throws IOException {
        inc c;
        try {
            InputStream in = new LowerCaseInputSteam(
                    new BufferedInputStream(
                            new FileInputStream(
                                    "test.txt")));
            while (c = in.read() >= 0) {
                System.out.print((char) c);
            }
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}