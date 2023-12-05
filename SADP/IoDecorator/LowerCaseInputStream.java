import java.io.IOException;
import java.io.InputStream;

public class LowerCaseInputStream extends InputStream {

    private InputStream inputStream;

    public LowerCaseInputStream(InputStream inputStream) {
        this.inputStream = inputStream;
    }

    @Override
    public int read() throws IOException {
        int value = inputStream.read();
        if (value >= 'A' && value <= 'Z') {
            value = value + 'a' - 'A';
        }
        return value;
    }
}