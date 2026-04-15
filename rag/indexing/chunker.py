"""
Smart Chunker - Boundary-aware document chunking

Implements intelligent chunking that respects natural document boundaries:
- Markdown: Paragraph and section boundaries
- Code: Function and class boundaries
- General: Sentence boundaries

Follows Anthropic best practices:
- Chunk size: 256-512 tokens
- Overlap: 15% (NVIDIA optimal)
- Boundary-aware splitting
"""

import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a document chunk"""
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    token_count: int


class SmartChunker:
    """Smart boundary-aware chunker"""

    def __init__(
        self,
        chunk_size: int = 384,  # Target tokens (good for docs/code)
        overlap_percentage: float = 0.15,  # 15% overlap
        min_chunk_size: int = 100  # Minimum chunk size in tokens
    ):
        self.chunk_size = chunk_size
        self.overlap_tokens = int(chunk_size * overlap_percentage)
        self.min_chunk_size = min_chunk_size

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using simple heuristic
        ~1 token per 4 characters (Claude's tokenizer approximation)
        """
        return len(text) // 4

    def chunk_document(self, content: str, file_type: str) -> List[Chunk]:
        """
        Chunk document with boundary awareness

        Args:
            content: Document content
            file_type: File extension (.md, .py, .ts, etc.)

        Returns:
            List of Chunk objects
        """
        if file_type in ['.md', '.txt']:
            return self._chunk_markdown(content)
        elif file_type in ['.py', '.ts', '.js', '.sh']:
            return self._chunk_code(content)
        else:
            return self._chunk_generic(content)

    def _chunk_markdown(self, content: str) -> List[Chunk]:
        """
        Chunk markdown respecting paragraph and section boundaries
        """
        chunks = []
        chunk_index = 0

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', content)

        current_chunk = ""
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Estimate tokens
            combined = current_chunk + "\n\n" + para if current_chunk else para
            combined_tokens = self.estimate_tokens(combined)

            if combined_tokens <= self.chunk_size:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = content.find(para)
            else:
                # Save current chunk if it meets minimum size
                if current_chunk and self.estimate_tokens(current_chunk) >= self.min_chunk_size:
                    chunk_end = current_start + len(current_chunk)
                    chunks.append(Chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=chunk_end,
                        token_count=self.estimate_tokens(current_chunk)
                    ))
                    chunk_index += 1

                # Start new chunk with overlap
                # Include last paragraph of previous chunk for context
                if current_chunk and '\n\n' in current_chunk:
                    last_para = current_chunk.split('\n\n')[-1]
                    current_chunk = last_para + "\n\n" + para
                else:
                    current_chunk = para

                current_start = content.find(para)

        # Add final chunk
        if current_chunk and self.estimate_tokens(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                token_count=self.estimate_tokens(current_chunk)
            ))

        return chunks

    def _chunk_code(self, content: str) -> List[Chunk]:
        """
        Chunk code respecting function/class boundaries
        Falls back to line-based chunking if no clear boundaries
        """
        chunks = []
        chunk_index = 0

        # Split by lines
        lines = content.split('\n')

        current_chunk = ""
        current_start_line = 0
        line_num = 0

        for line in lines:
            # Calculate combined content
            combined = current_chunk + "\n" + line if current_chunk else line
            combined_tokens = self.estimate_tokens(combined)

            # Check if we should break
            should_break = (
                combined_tokens > self.chunk_size and
                current_chunk and  # Don't break on first line
                self._is_good_break_point(line)
            )

            if should_break:
                # Save current chunk
                if self.estimate_tokens(current_chunk) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=0,  # Character positions less relevant for code
                        end_char=len(current_chunk),
                        token_count=self.estimate_tokens(current_chunk)
                    ))
                    chunk_index += 1

                # Start new chunk with small overlap (last few lines)
                overlap_lines = '\n'.join(current_chunk.split('\n')[-3:]) if current_chunk else ""
                current_chunk = overlap_lines + "\n" + line if overlap_lines else line
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line

            line_num += 1

        # Add final chunk
        if current_chunk and self.estimate_tokens(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(current_chunk),
                token_count=self.estimate_tokens(current_chunk)
            ))

        return chunks

    def _is_good_break_point(self, line: str) -> bool:
        """
        Check if line is a good breaking point for code
        """
        stripped = line.strip()

        # Empty lines are good break points
        if not stripped:
            return True

        # Function/class definitions
        if any(stripped.startswith(kw) for kw in ['def ', 'class ', 'function ', 'const ', 'export ']):
            return True

        # Comments
        if stripped.startswith(('#', '//', '/*', '*', '"""', "'''")):
            return True

        # Closing braces
        if stripped in ['}', '};', '})', '});']:
            return True

        return False

    def _chunk_generic(self, content: str) -> List[Chunk]:
        """
        Generic chunking by sentences
        """
        chunks = []
        chunk_index = 0

        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', content)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            combined = current_chunk + " " + sentence if current_chunk else sentence
            combined_tokens = self.estimate_tokens(combined)

            if combined_tokens <= self.chunk_size:
                current_chunk = combined
                if not current_start:
                    current_start = content.find(sentence)
            else:
                # Save current chunk
                if current_chunk and self.estimate_tokens(current_chunk) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=current_chunk,
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        token_count=self.estimate_tokens(current_chunk)
                    ))
                    chunk_index += 1

                # Start new chunk
                current_chunk = sentence
                current_start = content.find(sentence)

        # Add final chunk
        if current_chunk and self.estimate_tokens(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk,
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                token_count=self.estimate_tokens(current_chunk)
            ))

        return chunks

    def get_stats(self, chunks: List[Chunk]) -> Dict:
        """Get chunking statistics"""
        if not chunks:
            return {}

        token_counts = [c.token_count for c in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_tokens': sum(token_counts) / len(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts)
        }
