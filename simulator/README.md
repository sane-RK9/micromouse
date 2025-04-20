# Micromouse Simulator

This directory contains the Java-based simulator for visualizing the micromouse maze solving process.

## Setup

1. Compile the Java simulator:
```bash
javac MicromouseSimulator.java
```

2. Run the simulator:
```bash
java MicromouseSimulator
```

## Communication Protocol

The simulator communicates with the Python code via TCP socket on port 12345.

### Message Format
- Each message is a string in the format: `dx,dy\n`
  - `dx`: Change in x-direction (-1, 0, or 1)
  - `dy`: Change in y-direction (-1, 0, or 1)
  - `\n`: Newline terminator

### Error Handling
- The simulator will automatically handle connection errors
- Invalid moves will be ignored
- The simulator will maintain the mouse position within maze boundaries

## Features
- 16x16 maze visualization
- Real-time mouse position tracking
- Goal position display
- Interactive controls for maze size and mouse position 