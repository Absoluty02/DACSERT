import csv
import os
from dotenv import load_dotenv
import pandas as pd
from extractors import *
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.logging import RichHandler
from pydantic import BaseModel, Field, field_validator
from typeguard import typechecked
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dataclasses import dataclass
from enum import Enum

# Rich console and logging configuration
console = Console(force_terminal=True, force_interactive=True, color_system="truecolor")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
        enable_link_path=False
    )]
)
logger = logging.getLogger(__name__)

# Typer app
app = typer.Typer(
    name="LLM Orchestrator",
    help="🚀 Orchestrator to run parallel LLM calls using LangChain",
    add_completion=False
)


class PromptType(Enum):
    """Enum for different prompt types"""
    SYSTEM = "system"
    ENTRY = "entry"
    FIRST_JOB = "first_job"
    SECOND_JOB = "second_job"


@dataclass
class PromptTemplate:
    """Template for a prompt with metadata"""
    content: str
    prompt_type: PromptType
    file_path: Path
    variables: List[str] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = []

    @typechecked
    def format(self, **kwargs) -> str:
        """
        Format the prompt template with provided variables

        Args:
            **kwargs: Variables to format the template with

        Returns:
            Formatted prompt string
        """
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            logger.error(f"[red]Missing variable in prompt template: {e}[/red]")
            raise

    @typechecked
    def validate_variables(self, **kwargs) -> bool:
        """
        Validate that all required variables are provided

        Args:
            **kwargs: Variables to validate

        Returns:
            True if all variables are present
        """
        missing = [var for var in self.variables if var not in kwargs]
        if missing:
            logger.warning(f"[yellow]Missing variables: {missing}[/yellow]")
            return False
        return True


class PromptManager:
    """Manager for loading, validating, and managing prompts"""

    def __init__(self):
        """Initialize the prompt manager"""
        self.prompts: Dict[PromptType, PromptTemplate] = {}
        logger.info("[cyan]Prompt manager initialized[/cyan]")

    @typechecked
    def load_prompt(self, file_path: Path, prompt_type: PromptType,
                    variables: Optional[List[str]] = None) -> PromptTemplate:
        """
        Load a prompt from file

        Args:
            file_path: Path to the prompt file
            prompt_type: Type of prompt
            variables: List of variable names expected in the prompt

        Returns:
            PromptTemplate object

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        if not file_path.exists():
            logger.error(f"[red]Prompt file not found: {file_path}[/red]")
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        template = PromptTemplate(
            content=content,
            prompt_type=prompt_type,
            file_path=file_path,
            variables=variables or []
        )

        self.prompts[prompt_type] = template
        logger.info(f"[green]✓[/green] Loaded {prompt_type.value} prompt from {file_path}")

        return template

    @typechecked
    def load_all_prompts(self, system_path: Path, entry_path: Path,
                         first_job_path: Path, second_job_path: Path) -> None:
        """
        Load all prompts at once

        Args:
            system_path: Path to system prompt
            entry_path: Path to entry prompt
            first_job_path: Path to first job prompt
            second_job_path: Path to second job prompt
        """
        with console.status("[bold cyan]Loading prompts..."):
            self.load_prompt(system_path, PromptType.SYSTEM)
            self.load_prompt(entry_path, PromptType.ENTRY)
            self.load_prompt(first_job_path, PromptType.FIRST_JOB)
            self.load_prompt(second_job_path, PromptType.SECOND_JOB)

        console.print("[green]✓[/green] All prompts loaded successfully")

    @typechecked
    def get_prompt(self, prompt_type: PromptType) -> PromptTemplate:
        """
        Get a loaded prompt by type

        Args:
            prompt_type: Type of prompt to retrieve

        Returns:
            PromptTemplate object

        Raises:
            KeyError: If prompt hasn't been loaded
        """
        if prompt_type not in self.prompts:
            logger.error(f"[red]Prompt {prompt_type.value} not loaded[/red]")
            raise KeyError(f"Prompt {prompt_type.value} has not been loaded")

        return self.prompts[prompt_type]

    @typechecked
    def get_prompt_content(self, prompt_type: PromptType, **kwargs) -> str:
        """
        Get formatted prompt content

        Args:
            prompt_type: Type of prompt
            **kwargs: Variables to format the prompt with

        Returns:
            Formatted prompt string
        """
        template = self.get_prompt(prompt_type)

        if kwargs and template.variables:
            template.validate_variables(**kwargs)
            return template.format(**kwargs)

        return template.content

    @typechecked
    def reload_prompt(self, prompt_type: PromptType) -> PromptTemplate:
        """
        Reload a prompt from its file

        Args:
            prompt_type: Type of prompt to reload

        Returns:
            Reloaded PromptTemplate
        """
        if prompt_type not in self.prompts:
            raise KeyError(f"Prompt {prompt_type.value} has not been loaded yet")

        old_template = self.prompts[prompt_type]
        logger.info(f"[cyan]Reloading {prompt_type.value} prompt...[/cyan]")

        return self.load_prompt(
            old_template.file_path,
            prompt_type,
            old_template.variables
        )

    def display_prompts_info(self) -> None:
        """Display information about loaded prompts in a table"""
        table = Table(title="Loaded prompts")
        table.add_column("Type", style="cyan")
        table.add_column("File", style="yellow")
        table.add_column("Size (chars)", style="green")
        table.add_column("Variables", style="magenta")

        for prompt_type, template in self.prompts.items():
            variables = ", ".join(template.variables) if template.variables else "None"
            table.add_row(
                prompt_type.value,
                str(template.file_path.name),
                str(len(template.content)),
                variables
            )

        console.print(table)

    def validate_all_prompts(self) -> bool:
        """
        Validate that all required prompts are loaded

        Returns:
            True if all prompts are loaded
        """
        required_prompts = {PromptType.SYSTEM, PromptType.ENTRY,
                            PromptType.FIRST_JOB, PromptType.SECOND_JOB}
        loaded_prompts = set(self.prompts.keys())

        missing = required_prompts - loaded_prompts

        if missing:
            logger.error(f"[red]Missing prompts: {[p.value for p in missing]}[/red]")
            return False

        logger.info("[green]✓[/green] All required prompts are loaded")
        return True

    @typechecked
    def get_combined_prompt(self, json_content: str = "") -> Dict[str, str]:
        """
        Get all prompts combined for LLM processing

        Args:
            json_content: JSON content to append to entry prompt

        Returns:
            Dictionary with all prompt contents
        """
        if not self.validate_all_prompts():
            raise ValueError("Not all required prompts are loaded")

        return {
            "system": self.get_prompt_content(PromptType.SYSTEM),
            "entry": self.get_prompt_content(PromptType.ENTRY) + json_content,
            "first_job": self.get_prompt_content(PromptType.FIRST_JOB),
            "second_job": self.get_prompt_content(PromptType.SECOND_JOB)
        }


class OrchestratorConfig(BaseModel):
    """Validated configuration for the orchestrator"""

    gemini_model: str = Field(default="gemini-2.5-flash", description="Gemini model to use")
    gpt_model: str = Field(default="gpt-4o-mini", description="GPT model to use")
    claude_model: str = Field(default="claude-sonnet-4-20250514", description="Claude model to use")

    data_file: Path = Field(..., description="CSV file with data")
    output_file: Path = Field(..., description="Output CSV file")

    system_prompt_file: Path = Field(..., description="System prompt file")
    first_job_prompt_file: Path = Field(..., description="First job prompt file")
    second_job_prompt_file: Path = Field(..., description="Second job prompt file")
    entry_prompt_file: Path = Field(..., description="Entry prompt file")

    providers: List[str] = Field(default=["gemini", "gpt", "claude"], description="Providers to use")
    use_fallback: bool = Field(default=True, description="Use fallback if a provider fails")
    sample_size: int = Field(default=1, description="Number of samples to process")

    @field_validator('data_file', 'system_prompt_file', 'first_job_prompt_file',
                     'second_job_prompt_file', 'entry_prompt_file', mode='after')
    @classmethod
    def file_must_exist(cls, v):
        if not v.exists():
            raise ValueError(f"File {v} does not exist")
        return v

    @field_validator('providers', mode='after')
    @classmethod
    def validate_providers(cls, v):
        valid_providers = {"gemini", "gpt", "claude"}
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(f"Provider {provider} is not valid. Use: {valid_providers}")
        return v

    class Config:
        arbitrary_types_allowed = True


class LangChainLLMOrchestrator:
    """Orchestrator to run parallel LLM calls using LangChain"""

    @typechecked
    def __init__(self, config: OrchestratorConfig, prompt_manager: PromptManager):
        """
        Initialize the orchestrator with validated configuration and prompt manager

        Args:
            config: Validated configuration
            prompt_manager: Prompt manager instance
        """
        self.config = config
        self.prompt_manager = prompt_manager

        with console.status("[bold cyan]Initializing LLM clients..."):
            self.gemini_llm = ChatGoogleGenerativeAI(
                model=config.gemini_model,
                google_api_key=os.getenv("GEMINI_KEY"),
                temperature=0
            )

            self.gpt_llm = ChatOpenAI(
                model=config.gpt_model,
                api_key=os.getenv("OPENAI_KEY"),
                temperature=0
            )

            self.claude_llm = ChatAnthropic(
                model=config.claude_model,
                api_key=os.getenv("CLAUDE_KEY"),
                max_tokens=500,
                temperature=0
            )

            self.output_parser = StrOutputParser()

        console.print("[green]✓[/green] LLM clients initialized successfully")

    @typechecked
    def create_chain(self, llm, provider_name: str):
        """
        Create a LangChain chain for a specific LLM

        Args:
            llm: LangChain model instance
            provider_name: Provider name (for logging)

        Returns:
            Configured chain
        """

        def parse_with_error_handling(text):
            try:
                logger.info(f"[green]{provider_name}[/green] completed successfully")
                return manage_llm_response(text, provider_name.lower())
            except Exception as e:
                logger.error(f"[red]{provider_name}[/red] parsing error: {str(e)}")
                return {"error": str(e), "provider": provider_name.lower()}

        chain = llm | self.output_parser | parse_with_error_handling
        return chain

    @typechecked
    def create_messages(self, json_content: str = "") -> List:
        """
        Create message list for LLMs using prompt manager

        Args:
            json_content: JSON content to append to entry prompt

        Returns:
            Formatted message list
        """
        prompts = self.prompt_manager.get_combined_prompt(json_content)

        return [
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["entry"]),
            HumanMessage(content=prompts["first_job"]),
            HumanMessage(content=prompts["second_job"])
        ]

    @typechecked
    async def run_parallel(self, json_content: str = "",
                           providers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute LLM calls in parallel using LangChain

        Args:
            json_content: JSON content to append to entry prompt
            providers: List of providers to use

        Returns:
            Dict with results from each LLM
        """
        if providers is None:
            providers = self.config.providers

        messages = self.create_messages(json_content)

        chains_dict = {}

        if "gemini" in providers:
            chains_dict["gemini"] = self.create_chain(self.gemini_llm, "Gemini")

        if "gpt" in providers:
            chains_dict["gpt"] = self.create_chain(self.gpt_llm, "GPT")

        if "claude" in providers:
            chains_dict["claude"] = self.create_chain(self.claude_llm, "Claude")

        parallel_chain = RunnableParallel(**chains_dict)

        logger.info(f"[cyan]Starting parallel execution of {len(chains_dict)} LLMs[/cyan]")
        try:
            results = await parallel_chain.ainvoke(messages)
            logger.info("[green]✓ All LLM calls completed[/green]")
            return results
        except Exception as e:
            logger.error(f"[red]Parallel execution error: {str(e)}[/red]")
            return {provider: {"error": str(e), "provider": provider}
                    for provider in providers}

    @typechecked
    async def run_parallel_with_fallback(self, json_content: str = "",
                                         providers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute calls with fallback: if one provider fails, others continue

        Args:
            json_content: JSON content to append to entry prompt
            providers: List of providers to use

        Returns:
            Dict with results from each LLM
        """
        if providers is None:
            providers = self.config.providers

        messages = self.create_messages(json_content)

        async def safe_invoke(provider: str, chain):
            """Safe wrapper to invoke a chain"""
            try:
                logger.info(f"[cyan]Starting {provider} call[/cyan]")
                result = await chain.ainvoke(messages)
                return provider, result
            except Exception as e:
                logger.error(f"[red]{provider} error: {str(e)}[/red]")
                return provider, {"error": str(e), "provider": provider}

        tasks = []
        if "gemini" in providers:
            chain = self.create_chain(self.gemini_llm, "Gemini")
            tasks.append(safe_invoke("gemini", chain))

        if "gpt" in providers:
            chain = self.create_chain(self.gpt_llm, "GPT")
            tasks.append(safe_invoke("gpt", chain))

        if "claude" in providers:
            chain = self.create_chain(self.claude_llm, "Claude")
            tasks.append(safe_invoke("claude", chain))

        logger.info(f"[cyan]Starting parallel execution with fallback for {len(tasks)} LLMs[/cyan]")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[red]Task failed: {result}[/red]")
                continue
            provider, value = result
            result_dict[provider] = value

        logger.info("[green]✓ All calls with fallback completed[/green]")
        return result_dict


@typechecked
async def process_samples_parallel(orchestrator: LangChainLLMOrchestrator,
                                   df: pd.DataFrame, ids: set):
    """
    Process dataframe samples in parallel
    """
    config = orchestrator.config
    sample = df.sample(n=config.sample_size)

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Processing {len(sample)} samples...",
            total=len(sample)
        )

        for idx, row in sample.iterrows():
            if idx in ids:
                logger.info(f"[yellow]⊘ Skipping already processed: {idx}[/yellow]")
                progress.advance(task)
                continue

            json_content = row.to_json(orient="records")

            logger.info(f"[bold cyan]━━━ Processing index {idx} ━━━[/bold cyan]")

            if config.use_fallback:
                results = await orchestrator.run_parallel_with_fallback(
                    json_content=json_content,
                    providers=config.providers
                )
            else:
                results = await orchestrator.run_parallel(
                    json_content=json_content,
                    providers=config.providers
                )

            # Display results table
            table = Table(title=f"Results for index {idx}")
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")

            for provider, result in results.items():
                status = "✓ Success" if "error" not in result else f"✗ Error: {result.get('error', 'Unknown')}"
                table.add_row(provider.upper(), status)

            console.print(table)

            gemini_result = results.get("gemini", {})
            gpt_result = results.get("gpt", {})
            claude_result = results.get("claude", {})

            entry_row = new_entry_creator(
                index=idx,
                gpt_result=gpt_result,
                gemini_result=gemini_result,
                claude_result=claude_result
            )

            with open(config.output_file, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(entry_row)

            logger.info(f"[green]✓ Results saved for index {idx}[/green]")
            progress.advance(task)


@app.command()
def run(
        data_file: Path = typer.Option("dataset/v4_atomic_all.csv", "--data", "-d", help="CSV file with data"),
        output_file: Path = typer.Option("responses/llm_responses.csv", "--output", "-o", help="Output CSV file"),
        system_prompt: Path = typer.Option("prompts/system_prompt", "--system", "-s", help="System prompt file"),
        first_job: Path = typer.Option("prompts/first_job_prompt", "--first", "-f", help="First job prompt file"),
        second_job: Path = typer.Option("prompts/second_job_prompt", "--second", "-j", help="Second job prompt file"),
        entry: Path = typer.Option("prompts/entry_prompt", "--entry", "-e", help="Entry prompt file"),
        providers: List[str] = typer.Option(["gemini", "gpt", "claude"], "--provider", "-p", help="Providers to use"),
        fallback: bool = typer.Option(True, "--fallback/--no-fallback", help="Use fallback"),
        sample_size: int = typer.Option(1, "--samples", "-n", help="Number of samples to process"),
        gemini_model: str = typer.Option("gemini-2.5-flash", "--gemini-model", help="Gemini model"),
        gpt_model: str = typer.Option("gpt-4o-mini", "--gpt-model", help="GPT model"),
        claude_model: str = typer.Option("claude-sonnet-4-20250514", "--claude-model", help="Claude model"),
):
    """
    🚀 Run the parallel LLM orchestrator
    """
    console.print(Panel.fit(
        "[bold cyan]LLM Orchestrator[/bold cyan]\n"
        "[dim]Parallel execution of Gemini, GPT and Claude[/dim]",
        border_style="cyan"
    ))

    try:
        # Configuration validation
        config = OrchestratorConfig(
            gemini_model=gemini_model,
            gpt_model=gpt_model,
            claude_model=claude_model,
            data_file=data_file,
            output_file=output_file,
            system_prompt_file=system_prompt,
            first_job_prompt_file=first_job,
            second_job_prompt_file=second_job,
            entry_prompt_file=entry,
            providers=providers,
            use_fallback=fallback,
            sample_size=sample_size
        )

        console.print("[green]✓[/green] Configuration validated")

        # Display configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Gemini Model", config.gemini_model)
        config_table.add_row("GPT Model", config.gpt_model)
        config_table.add_row("Claude Model", config.claude_model)
        config_table.add_row("Providers", ", ".join(config.providers))
        config_table.add_row("Fallback", "Yes" if config.use_fallback else "No")
        config_table.add_row("Sample Size", str(config.sample_size))

        console.print(config_table)

        asyncio.run(run_async(config))

        console.print(Panel.fit(
            "[bold green]✓ Execution completed successfully![/bold green]",
            border_style="green"
        ))

    except Exception as e:
        console.print(f"[bold red]✗ Error: {str(e)}[/bold red]")
        raise typer.Exit(code=1)


async def run_async(config: OrchestratorConfig):
    """Main async function"""
    load_dotenv()

    # Initialize prompt manager and load all prompts
    prompt_manager = PromptManager()
    prompt_manager.load_all_prompts(
        system_path=config.system_prompt_file,
        entry_path=config.entry_prompt_file,
        first_job_path=config.first_job_prompt_file,
        second_job_path=config.second_job_prompt_file
    )

    # Display loaded prompts info
    prompt_manager.display_prompts_info()

    # Initialize orchestrator with prompt manager
    orchestrator = LangChainLLMOrchestrator(config=config, prompt_manager=prompt_manager)

    # Load data
    with console.status("[bold cyan]Loading data..."):
        df = pd.read_csv(config.data_file)
        console.print(f"[green]✓[/green] Loaded {len(df)} records from {config.data_file}")

    ids = set()
    try:
        with open(config.output_file, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.add(row['id'])
        console.print(f"[yellow]⚠[/yellow] Found {len(ids)} already processed records")
    except FileNotFoundError:
        console.print(f"[yellow]⚠[/yellow] {config.output_file} not found, starting fresh")

    # Process samples
    await process_samples_parallel(
        orchestrator=orchestrator,
        df=df,
        ids=ids
    )


if __name__ == "__main__":
    app()