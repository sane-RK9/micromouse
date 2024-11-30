import java.awt.*;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import javax.swing.*;

public class MicromouseSimulator extends JFrame {
    private static final int MAZE_SIZE = 16;
    private static final int CELL_SIZE = 30;

    private int mouseX = 0;
    private int mouseY = 0;
    private int goalX = MAZE_SIZE - 1;
    private int goalY = MAZE_SIZE - 1;
    private boolean displayMouseAtCenter = true;

    private boolean[][] maze;

    private JPanel mazePanel;
    private JButton increaseMazeSize, decreaseMazeSize;
    private JButton increaseMouseX, decreaseMouseX, increaseMouseY, decreaseMouseY;
    private JButton toggleGoalDisplay;

    private ServerSocket serverSocket;
    private Socket clientSocket;
    private DataInputStream in;
    private DataOutputStream out;

    public MicromouseSimulator() {
        setTitle("Micromouse Simulator");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Initialize the maze
        maze = new boolean[MAZE_SIZE][MAZE_SIZE];
        // Populate the maze with walls and free space

        // Maze panel
        mazePanel = new MazePanel();
        mazePanel.setPreferredSize(new Dimension(MAZE_SIZE * CELL_SIZE, MAZE_SIZE * CELL_SIZE));
        add(mazePanel, BorderLayout.CENTER);

        // Control panel
        JPanel controlPanel = new JPanel(new GridLayout(2, 4, 10, 10));
        add(controlPanel, BorderLayout.SOUTH);

        // Maze size controls
        increaseMazeSize = new JButton("+");
        decreaseMazeSize = new JButton("-");
        controlPanel.add(decreaseMazeSize);
        controlPanel.add(increaseMazeSize);

        // Mouse position controls
        increaseMouseX = new JButton("→");
        decreaseMouseX = new JButton("←");
        increaseMouseY = new JButton("↓");
        decreaseMouseY = new JButton("↑");
        controlPanel.add(decreaseMouseX);
        controlPanel.add(increaseMouseX);
        controlPanel.add(decreaseMouseY);
        controlPanel.add(increaseMouseY);

        // Goal display toggle
        toggleGoalDisplay = new JButton("Toggle Goal");
        controlPanel.add(toggleGoalDisplay);

        pack();
        setVisible(true);

        // Set up the server socket to listen for connections from Python
        try {
            serverSocket = new ServerSocket(12345);
            System.out.println("Waiting for connection from Python...");
            clientSocket = serverSocket.accept();
            System.out.println("Connection established.");

            in = new DataInputStream(clientSocket.getInputStream());
            out = new DataOutputStream(clientSocket.getOutputStream());

            // Handle commands from Python
            handlePythonCommands();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void handlePythonCommands() {
        try {
            while (true) {
                byte command = in.readByte();
                if (command == 0) {
                    // Python is requesting the maze state
                    sendMazeState(out);
                } else if (command == 1) {
                    // Python is sending a path
                    receivePath(in);
                    mazePanel.repaint();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private boolean isCellTraversable(int x, int y) {
        return x >= 0 && x < MAZE_SIZE && y >= 0 && y < MAZE_SIZE && !maze[x][y];
    }

    private void sendMazeState(OutputStream outputStream) {
        try {
            DataOutputStream dos = new DataOutputStream(outputStream);
            dos.writeInt(MAZE_SIZE);
            for (int x = 0; x < MAZE_SIZE; x++) {
                for (int y = 0; y < MAZE_SIZE; y++) {
                    dos.writeBoolean(maze[x][y]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void receivePath(InputStream inputStream) {
        try {
            DataInputStream dis = new DataInputStream(inputStream);
            int pathLength = dis.readInt();
            for (int i = 0; i < pathLength; i++) {
                int dx = dis.readInt();
                int dy = dis.readInt();
                updateMousePosition(dx, dy);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void updateMousePosition(int dx, int dy) {
        mouseX = Math.max(0, Math.min(MAZE_SIZE - 1, mouseX + dx));
        mouseY = Math.max(0, Math.min(MAZE_SIZE - 1, mouseY + dy));
    }

    private class MazePanel extends JPanel {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            drawMaze(g);
        }
    }

    private void drawMaze(Graphics g) {
        // Draw the maze grid
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, MAZE_SIZE * CELL_SIZE, MAZE_SIZE * CELL_SIZE);
        g.setColor(Color.BLACK);
        for (int i = 0; i <= MAZE_SIZE; i++) {
            g.drawLine(i * CELL_SIZE, 0, i * CELL_SIZE, MAZE_SIZE * CELL_SIZE); // Vertical lines
            g.drawLine(0, i * CELL_SIZE, MAZE_SIZE * CELL_SIZE, i * CELL_SIZE); // Horizontal lines
        }

        // Draw the micromouse
        g.setColor(Color.RED);
        int mouseDrawX = displayMouseAtCenter ? mouseX * CELL_SIZE + CELL_SIZE / 2 - 10 : mouseX * CELL_SIZE + 5;
        int mouseDrawY = displayMouseAtCenter ? mouseY * CELL_SIZE + CELL_SIZE / 2 - 10 : mouseY * CELL_SIZE + 5;
        g.fillOval(mouseDrawX, mouseDrawY, CELL_SIZE - 10, CELL_SIZE - 10);

        // Draw the goal
        g.setColor(Color.GREEN);
        int goalDrawX = goalX * CELL_SIZE + CELL_SIZE / 2 - 10;
        int goalDrawY = goalY * CELL_SIZE + CELL_SIZE / 2 - 10;
        g.fillOval(goalDrawX, goalDrawY, CELL_SIZE - 10, CELL_SIZE - 10);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new MicromouseSimulator());
    }
}