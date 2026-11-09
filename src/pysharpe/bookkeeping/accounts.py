"""Account identity for portfolio bookkeeping.

A Canadian investor normally holds several accounts, often more than one of the
same registration type — two TFSAs at different brokers, a cash and a margin
non-registered account.  :class:`Account` separates two facts that are commonly
conflated:

* ``id`` identifies *which* account a trade happened in (a stable, caller-chosen
  label such as ``"TFSA-2024"``);
* ``registration`` is the tax wrapper (:class:`AccountType`), which decides how
the account is taxed;
* ``owner`` names the taxpayer the account belongs to.

Registration drives tax treatment, identity drives bookkeeping.  Only
non-registered accounts (:attr:`Account.is_taxable`) join the pooled adjusted
cost base, because that pool is computed per taxpayer rather than per account
(ITA s. 47(1)).

When a journal supplies a bare :class:`AccountType` instead of an
:class:`Account`, it refers to the single default account of that registration
(:meth:`Account.from_registration`).  That keeps historical journals working
without granting them the ability to merge two real accounts.
"""

from __future__ import annotations

from dataclasses import dataclass

from pysharpe.exceptions import DataValidationError
from pysharpe.optimization.tax_location import AccountType

#: Owner label used when the caller does not distinguish taxpayers.
SELF_OWNER = "self"


def coerce_registration(value: object) -> AccountType:
    """Resolve *value* to an :class:`AccountType` member.

    Only declared member names are accepted.  Alias labels (``"Non-Reg"``,
    ``"Taxable"``, …) are rejected instead of silently falling back to
    non-registered, because that fallback would change the account's tax
    treatment without warning.
    """

    if isinstance(value, AccountType):
        return value
    label = str(value).strip().upper()
    try:
        return AccountType(label)
    except ValueError as exc:
        choices = ", ".join(member.value for member in AccountType)
        raise DataValidationError(
            f"account must be an AccountType or one of {choices}; got {value!r}"
        ) from exc


def _non_empty(label: str, value: object) -> str:
    """Return *value* as a trimmed, non-empty string."""

    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{label} must be a non-empty string; got {value!r}")
    return value.strip()


@dataclass(frozen=True)
class Account:
    """One brokerage account belonging to one taxpayer.

    Attributes:
        id (str): Stable identifier for the account, chosen by the caller
            (``"TFSA-2024"``, ``"RRSP-work"``).  Two accounts of the same
            registration must use different ids.
        registration (AccountType): Tax wrapper of the account.
        owner (str): Taxpayer the account belongs to.  A
            :class:`~pysharpe.bookkeeping.PortfolioLedger` is scoped to one
            owner and rejects foreign accounts rather than pooling them.

    Raises:
        DataValidationError: If ``id`` or ``owner`` is empty, or the
            registration is not a declared :class:`AccountType`.

    Example:
        >>> from pysharpe.bookkeeping import Account
        >>> from pysharpe.optimization.tax_location import AccountType
        >>> account = Account("TFSA-2024", AccountType.TFSA)
        >>> account.is_taxable
        False
        >>> Account("margin", AccountType.NON_REG).is_taxable
        True
    """

    id: str
    registration: AccountType
    owner: str = SELF_OWNER

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _non_empty("account id", self.id))
        object.__setattr__(self, "registration", coerce_registration(self.registration))
        object.__setattr__(self, "owner", _non_empty("account owner", self.owner))

    @property
    def is_taxable(self) -> bool:
        """True for non-registered accounts, which carry a taxable ACB.

        Registered plans (TFSA, FHSA, RRSP, LIRA, RRIF) generate no capital
        gains to report, so their cost base is book value for reporting only and
        never joins the pooled taxable ACB.
        """

        return self.registration is AccountType.NON_REG

    @classmethod
    def from_registration(
        cls, registration: object, *, owner: str = SELF_OWNER
    ) -> Account:
        """Return the single default account for a registration type.

        This is the account a bare ``AccountType`` refers to when a journal does
        not distinguish individual accounts.  Its id is the registration label
        (``"TFSA"``, ``"NON_REG"``).

        Example:
            >>> from pysharpe.bookkeeping import Account
            >>> from pysharpe.optimization.tax_location import AccountType
            >>> Account.from_registration(AccountType.NON_REG).id
            'NON_REG'
        """

        resolved = coerce_registration(registration)
        return cls(id=resolved.value, registration=resolved, owner=owner)


def account_of(value: object, *, owner: str = SELF_OWNER) -> Account:
    """Normalize *value* to an :class:`Account`.

    An existing :class:`Account` is returned unchanged (including its owner); a
    bare registration collapses to :meth:`Account.from_registration`.  Any other
    value is rejected by :func:`coerce_registration`.
    """

    if isinstance(value, Account):
        return value
    return Account.from_registration(value, owner=owner)
