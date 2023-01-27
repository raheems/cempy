"""Console script for cempy."""
import sys
import cl


@click.command()
def main(args=None):
    """Console script for cempy."""
    click.echo("Replace this message by putting your code into "
               "cempy.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
