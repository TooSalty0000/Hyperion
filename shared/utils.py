"""Shared utilities."""

import re


class VirtualScreen:
    """
    A simple terminal emulator that maintains a buffer of lines
    and handles basic cursor movements (Up, Backspace, Carriage Return).
    """
    def __init__(self):
        self.lines = [""]
        self.row = 0
        self.col = 0

    def write(self, text: str):
        i = 0
        while i < len(text):
            char = text[i]

            # Check for CSI sequences (Control Sequence Introducer) starts with \x1b[
            if char == '\x1b' and i+1 < len(text) and text[i+1] == '[':
                j = i + 2
                while j < len(text) and not ('@' <= text[j] <= '~'):
                    j += 1

                if j < len(text):
                    code = text[j]

                    if code == 'A':  # Cursor Up
                         if self.row > 0:
                             self.row -= 1
                    elif code == 'B':  # Cursor Down
                         self.row += 1
                         self.ensure_row()
                    elif code == 'K':  # Erase in line
                         current_line = self.lines[self.row]
                         if self.col < len(current_line):
                              self.lines[self.row] = current_line[:self.col]

                    i = j + 1
                    continue

            # Simple Control Chars
            if char == '\r':
                self.col = 0
            elif char == '\n':
                self.row += 1
                self.col = 0
                self.ensure_row()
            elif char == '\x08':  # Backspace
                if self.col > 0:
                    self.col -= 1
            else:
                 # Normal char
                 self.ensure_row()
                 line = self.lines[self.row]

                 if self.col > len(line):
                      line += ' ' * (self.col - len(line))

                 if self.col < len(line):
                      self.lines[self.row] = line[:self.col] + char + line[self.col+1:]
                 else:
                      self.lines[self.row] += char
                 self.col += 1

            i += 1

    def ensure_row(self):
        while len(self.lines) <= self.row:
             self.lines.append("")

    def get_text(self):
         return "\n".join(self.lines)

    @staticmethod
    def clean_ansi_colors(text):
        """Removes color/style codes and focus sequences, preserves cursor codes."""
        # Remove color/style codes: \x1b[...m
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        # Remove focus in/out sequences: \x1b[I and \x1b[O
        text = re.sub(r'\x1b\[[IO]', '', text)
        # Remove bracketed paste mode sequences: \x1b[?2004h and \x1b[?2004l
        text = re.sub(r'\x1b\[\?2004[hl]', '', text)
        return text

    @staticmethod
    def clean_all_ansi_simple(text):
        """Legacy fully strip for safety in portions."""
        return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
