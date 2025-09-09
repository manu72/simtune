from collections import Counter
from typing import Any, Dict

from rich.console import Console
from rich.table import Table

from core.models import Dataset

console = Console()


class DatasetValidator:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.validation_results: dict[str, Any] = {}

    def validate(self) -> Dict[str, Any]:
        console.print("\n[bold blue]Validating Dataset[/bold blue]")

        results = {
            "size_check": self._check_size(),
            "format_check": self._check_format(),
            "content_quality": self._check_content_quality(),
            "diversity_check": self._check_diversity(),
            "safety_check": self._check_safety(),
        }

        self.validation_results = results
        self._display_results(results)

        return results

    def _check_size(self) -> Dict[str, Any]:
        size = self.dataset.size

        if size == 0:
            status = "error"
            message = "Dataset is empty"
            recommendation = "Add training examples to the dataset"
        elif size < 10:
            status = "warning"
            message = f"Dataset has only {size} examples"
            recommendation = (
                "Add more examples (recommended: 10-100) for better fine-tuning"
            )
        elif size < 100:
            status = "good"
            message = f"Dataset has {size} examples - good for basic fine-tuning"
            recommendation = "Consider adding more examples for better results"
        else:
            status = "excellent"
            message = f"Dataset has {size} examples - excellent for fine-tuning"
            recommendation = "Dataset size is optimal"

        return {
            "status": status,
            "message": message,
            "recommendation": recommendation,
            "count": size,
        }

    def _check_format(self) -> Dict[str, Any]:
        issues = []

        for i, example in enumerate(self.dataset.examples):
            try:
                # Check required structure
                if not example.messages:
                    issues.append(f"Example {i+1}: No messages found")
                    continue

                # Check message roles
                roles = [msg.get("role") for msg in example.messages]
                if "user" not in roles:
                    issues.append(f"Example {i+1}: Missing user message")
                if "assistant" not in roles:
                    issues.append(f"Example {i+1}: Missing assistant message")

                # Check message content
                for j, msg in enumerate(example.messages):
                    if not msg.get("content", "").strip():
                        issues.append(f"Example {i+1}, Message {j+1}: Empty content")

            except Exception as e:
                issues.append(f"Example {i+1}: Format error - {str(e)}")

        if not issues:
            return {
                "status": "excellent",
                "message": "All examples are properly formatted",
                "recommendation": "Format validation passed",
                "issues": [],
            }
        elif (
            len(issues) < len(self.dataset.examples) * 0.1
        ):  # Less than 10% have issues
            return {
                "status": "warning",
                "message": f"{len(issues)} formatting issues found",
                "recommendation": "Fix formatting issues for better results",
                "issues": issues[:5],  # Show first 5 issues
            }
        else:
            return {
                "status": "error",
                "message": f"Significant formatting issues ({len(issues)} errors)",
                "recommendation": "Fix formatting issues before fine-tuning",
                "issues": issues[:10],  # Show first 10 issues
            }

    def _check_content_quality(self) -> Dict[str, Any]:
        if not self.dataset.examples:
            return {"status": "error", "message": "No examples to analyze"}

        stats = {
            "avg_prompt_length": 0.0,
            "avg_response_length": 0.0,
            "too_short_responses": 0,
            "too_long_responses": 0,
            "empty_responses": 0,
        }

        prompt_lengths = []
        response_lengths = []

        for example in self.dataset.examples:
            user_msg = next(
                (msg["content"] for msg in example.messages if msg["role"] == "user"),
                "",
            )
            assistant_msg = next(
                (
                    msg["content"]
                    for msg in example.messages
                    if msg["role"] == "assistant"
                ),
                "",
            )

            prompt_len = len(user_msg.split())
            response_len = len(assistant_msg.split())

            prompt_lengths.append(prompt_len)
            response_lengths.append(response_len)

            if response_len == 0:
                stats["empty_responses"] += 1
            elif response_len < 10:
                stats["too_short_responses"] += 1
            elif response_len > 1000:
                stats["too_long_responses"] += 1

        stats["avg_prompt_length"] = sum(prompt_lengths) / len(prompt_lengths)
        stats["avg_response_length"] = sum(response_lengths) / len(response_lengths)

        issues = []
        if stats["empty_responses"] > 0:
            issues.append(f"{stats['empty_responses']} empty responses")
        if stats["too_short_responses"] > len(self.dataset.examples) * 0.3:
            issues.append(
                f"{stats['too_short_responses']} very short responses (< 10 words)"
            )
        if stats["too_long_responses"] > 0:
            issues.append(
                f"{stats['too_long_responses']} very long responses (> 1000 words)"
            )

        if not issues:
            status = "good"
            message = f"Content quality looks good (avg: {stats['avg_response_length']:.0f} words per response)"
        elif len(issues) <= 2:
            status = "warning"
            message = f"Some content quality issues: {', '.join(issues)}"
        else:
            status = "error"
            message = f"Multiple content quality issues: {', '.join(issues)}"

        return {
            "status": status,
            "message": message,
            "recommendation": (
                "Review and improve short or empty responses"
                if issues
                else "Content quality is acceptable"
            ),
            "stats": stats,
        }

    def _check_diversity(self) -> Dict[str, Any]:
        if not self.dataset.examples:
            return {"status": "error", "message": "No examples to analyze"}

        prompts = []
        responses = []

        for example in self.dataset.examples:
            user_msg = next(
                (msg["content"] for msg in example.messages if msg["role"] == "user"),
                "",
            )
            assistant_msg = next(
                (
                    msg["content"]
                    for msg in example.messages
                    if msg["role"] == "assistant"
                ),
                "",
            )

            prompts.append(user_msg.lower())
            responses.append(assistant_msg.lower())

        # Check for duplicate or very similar prompts
        prompt_counter = Counter(prompts)
        duplicates = sum(1 for count in prompt_counter.values() if count > 1)

        # Check for response diversity (simple word overlap check)
        response_words = [set(resp.split()) for resp in responses]
        high_overlap_pairs = 0

        for i in range(len(response_words)):
            for j in range(
                i + 1, min(i + 10, len(response_words))
            ):  # Check against next 10 responses
                if len(response_words[i]) > 0 and len(response_words[j]) > 0:
                    overlap = len(response_words[i] & response_words[j]) / len(
                        response_words[i] | response_words[j]
                    )
                    if overlap > 0.7:  # 70% word overlap
                        high_overlap_pairs += 1

        issues = []
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate prompts")
        if high_overlap_pairs > len(self.dataset.examples) * 0.2:
            issues.append(f"{high_overlap_pairs} very similar responses")

        if not issues:
            status = "good"
            message = "Good diversity in prompts and responses"
        elif len(issues) == 1 and duplicates < len(self.dataset.examples) * 0.2:
            status = "warning"
            message = f"Some diversity issues: {', '.join(issues)}"
        else:
            status = "warning"
            message = f"Diversity concerns: {', '.join(issues)}"

        return {
            "status": status,
            "message": message,
            "recommendation": (
                "Add more varied prompts and responses"
                if issues
                else "Diversity looks good"
            ),
        }

    def _check_safety(self) -> Dict[str, Any]:
        # Basic safety checks for problematic content
        safety_keywords = [
            "suicide",
            "bomb",
        ]

        flagged_examples = []

        for i, example in enumerate(self.dataset.examples):
            content = " ".join([msg["content"].lower() for msg in example.messages])

            for keyword in safety_keywords:
                if keyword in content:
                    flagged_examples.append((i + 1, keyword))
                    break

        if not flagged_examples:
            return {
                "status": "good",
                "message": "No obvious safety concerns detected",
                "recommendation": "Content appears safe for fine-tuning",
            }
        elif len(flagged_examples) < len(self.dataset.examples) * 0.1:
            return {
                "status": "warning",
                "message": f"{len(flagged_examples)} examples may contain sensitive content",
                "recommendation": "Review flagged examples manually",
                "flagged": flagged_examples[:5],
            }
        else:
            return {
                "status": "error",
                "message": f"Multiple examples ({len(flagged_examples)}) contain potentially unsafe content",
                "recommendation": "Remove or modify unsafe content before fine-tuning",
                "flagged": flagged_examples[:10],
            }

    def _display_results(self, results: Dict[str, Any]) -> None:
        console.print("\n[bold]Validation Results[/bold]")

        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="white", max_width=60)

        status_colors = {
            "excellent": "green",
            "good": "green",
            "warning": "yellow",
            "error": "red",
        }

        for check_name, result in results.items():
            status = result.get("status", "unknown")
            color = status_colors.get(status, "white")
            status_display = f"[{color}]{status.upper()}[/{color}]"

            details = result.get("message", "No details")

            table.add_row(check_name.replace("_", " ").title(), status_display, details)

        console.print(table)

        # Show recommendations
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for check_name, result in results.items():
            rec = result.get("recommendation", "")
            if rec:
                console.print(f"• {rec}")

        # Show overall status
        statuses = [result.get("status") for result in results.values()]
        if "error" in statuses:
            overall = "[red]NEEDS ATTENTION[/red]"
        elif "warning" in statuses:
            overall = "[yellow]ACCEPTABLE WITH CAUTION[/yellow]"
        else:
            overall = "[green]READY FOR FINE-TUNING[/green]"

        console.print(f"\n[bold]Overall Status: {overall}[/bold]")

    def get_validation_summary(self) -> str:
        if not self.validation_results:
            return "Dataset not validated"

        statuses = [result.get("status") for result in self.validation_results.values()]
        if "error" in statuses:
            return "needs_attention"
        elif "warning" in statuses:
            return "acceptable"
        else:
            return "ready"
