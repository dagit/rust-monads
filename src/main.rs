use ghost::phantom;
use std::rc::Rc;

fn main() {
    println!("Hello, world!");
    let a = Cont::<i32, i32>::pure(1);
    println!("a = {}", a.clone().eval_cont());
    let b = a.fmap(|x| x + 1);
    println!("b = {}", b.clone().run_cont(|x| x));
    let c = b.bind(|x: i32| Cont::pure(x + x));
    println!("c = {}", c.run_cont(|x| x));
    let a = Cont::<i32, i32>::pure(1);
    let b = a.fmap(|x| x + 1);
    println!("b = {}", b.clone().eval_cont());
    let a = Cont::<i32, i32>::pure(1);
    let b = a.bind(|x: i32| Cont::pure(x + x));
    println!("b = {}", b.eval_cont());
    let c = call_cc(|exit1| {
        Cont::pure(1)
            .bind(move |_: i32| exit1('a'))
            .bind(|_: i32| unimplemented!())
    });
    println!("c = {:#?}", c.eval_cont());
    // We can also do a nested exit
    let c = call_cc(|exit1| -> Cont<'_, char, char> {
        let exit1_clone = exit1.clone();
        call_cc(move |_: Rc<dyn Fn(char) -> Cont<'static, char, char>>| exit1('c'))
            .bind(move |_: char| exit1_clone('b'))
    });
    println!("c = {:#?}", c.eval_cont());

    let c = mdo! {
        pure 1;
    };
    println!("c = {:#?}", c.eval_cont());
    let c: Cont<'_, i32, i32> = mdo! {
        a <- pure 1;
        pure a;
    };
    println!("c = {:#?}", c.eval_cont());
    ex10();
}

#[test]
fn ex1() {
    let ex = mdo! {
        a <- pure 1;
        b <- pure 10;
        pure (a+b);
    };
    println!("{}", ex.run_cont(|x| { format!("{:#?}", x) }));
}

#[test]
fn ex2() {
    let ex = mdo! {
        a <- pure 1;
        b <- Cont::cont(|fred| fred(10));
        pure (a+b);
    };
    println!("{}", ex.run_cont(|x| { format!("{:#?}", x) }));
}

#[test]
fn ex3() {
    let ex = mdo! {
        a <- pure 1;
        b <- Cont::cont(|_fred: Rc<dyn Fn(i32) -> _>| "escape".into());
        pure (a+b);
    };
    println!("{}", ex.run_cont(|x| { format!("{:#?}", x) }));
}

#[test]
fn ex4() {
    let ex = mdo! {
        a <- pure 1;
        b <- Cont::cont(|fred: Rc<dyn Fn(i32) -> String>| {
            fred(10) + &fred(20)
        });
        pure (a+b);
    };
    println!("{}", ex.run_cont(|x| { format!("{:#?}", x) }));
}

#[test]
fn ex6() {
    let ex = mdo! {
        a <- pure 1;
        b <- Cont::cont(|fred: Rc<dyn Fn(i32) -> Vec<i32>>| {
            let mut f = fred(10);
            f.extend(fred(20).into_iter());
            f
        });
        pure (a+b);
    };
    println!("{}", DisplayVec(ex.run_cont(|x| vec![x])));
}

#[test]
fn ex8() {
    let ex = mdo! {
        a <- pure 1;
        b <- Cont::cont(|fred: Rc<dyn Fn(i32) -> Vec<i32>>| {
            VecFamily::bind(vec![10,20], &*fred)
        });
        pure (a+b);
    };
    println!("{}", DisplayVec(ex.run_cont(|x| vec![x])));
}

#[test]
fn ex9() {
    let ex: Cont<'_, Vec<i32>, i32> = mdo! {
        (a:i32) <- lift(VecFamily, vec![1,2]);
        b <- lift(VecFamily, vec![10,20]);
        pure (a+b);
    };
    println!("{}", DisplayVec::<i32>(run(VecFamily, ex)));
}

fn ex10() {
    // TODO: calling this identity is actually really weird...
    let ex: Cont<'_, Identity<()>, ()> = mdo! {
        lift_ii(pure_i(println!("What is your name?")));
        name <- lift_ii(pure_i(get_line()));
        lift_ii(pure_i(println!("Merry Xmas {}", name)));
        pure ();
    };
    println!("{}", run(IdentityFamily, ex));
}

pub fn get_line() -> String {
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        Ok(_) => input,
        Err(_) => panic!(),
    }
}

#[derive(Debug)]
struct DisplayVec<A>(pub Vec<A>);

impl<A> std::fmt::Display for DisplayVec<A>
where
    A: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let elements = self
            .0
            .iter()
            .map(|x| format!("{:#?}", x))
            .collect::<Vec<String>>()
            .join(", ");
        writeln!(f, "[{}]", elements)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct Cont<'a, R, A> {
    //run: (a -> r) -> r
    pub run: Rc<dyn Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a>,
}

impl<'a, R, A> Cont<'a, R, A> {
    ///cont :: ((a -> r) -> r) -> Cont r a
    pub fn cont<F>(f: F) -> Cont<'a, R, A>
    where
        F: Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a,
    {
        Cont { run: Rc::new(f) }
    }
    pub fn fmap<B, F>(self, f: F) -> Cont<'a, R, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        A: 'a,
        B: 'a,
        R: 'a,
    {
        //fmap f m = Cont $ \c -> run (c . f)
        //         = Cont (\c -> run (c . f))
        //         = Cont (\c -> run (\a -> c (f a)))
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R + 'a>| {
                let f = f.clone();
                (self.run)(Rc::new(move |a: A| c(f(a))))
            }),
        }
    }
    pub fn pure(a: A) -> Cont<'a, R, A>
    where
        A: 'a + Clone,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(A) -> R>| c(a.clone())),
        }
    }
    pub fn bind<B, K>(self, k: K) -> Cont<'a, R, B>
    where
        K: Fn(A) -> Cont<'a, R, B> + 'a + Clone,
        A: 'a,
        R: 'a,
        B: 'a,
    {
        Cont {
            run: Rc::new(move |c: Rc<dyn Fn(B) -> R>| {
                let k = k.clone();
                (self.run)(Rc::new(move |a: A| (k(a).run)(c.clone())))
            }),
        }
    }
    pub fn run_cont<F>(self, f: F) -> R
    where
        F: Fn(A) -> R + 'a,
    {
        (self.run)(Rc::new(f))
    }
}
impl<'a, R> Cont<'a, R, R> {
    pub fn eval_cont(self) -> R {
        self.run_cont(|r: R| r)
    }
}

pub fn call_cc<'a, A, B, R, F>(f: F) -> Cont<'a, R, A>
where
    F: for<'b> Fn(Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>) -> Cont<'a, R, A> + 'a,
    R: 'a + Clone,
    A: 'a + Clone,
{
    /* callCC f = ContT $ \ c -> runContT (f (\ x -> ContT $ \ _ -> c x)) c */
    let runit = call_cc_outer(f);
    Cont { run: runit }
}

fn call_cc_outer<'a, A, B, R, F>(f: F) -> Rc<dyn for<'b> Fn(Rc<dyn Fn(A) -> R + 'a>) -> R + 'a>
where
    F: for<'b> Fn(Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>) -> Cont<'a, R, A> + 'a,
    A: Clone + 'a,
    R: 'a,
{
    Rc::new(move |c| {
        let inner = call_cc_inner(c.clone());
        (f(inner).run)(c)
    })
}

fn call_cc_inner<'a, A, B, R>(c: Rc<dyn Fn(A) -> R + 'a>) -> Rc<dyn Fn(A) -> Cont<'a, R, B> + 'a>
where
    A: 'a + Clone,
    R: 'a,
{
    Rc::new(move |a: A| {
        let c = c.clone();
        Cont {
            run: Rc::new(move |_| c(a.clone())),
        }
    })
}

// http://blog.sigfpe.com/2008/12/mother-of-all-monads.html
// i x = cont (\fred -> x >>= fred)
// where cont f = Cont { run: f }
// i :: Monad m => m a -> Cont (m b) a
pub fn lift<'a, A, B, M>(_: M, m: This<M, A>) -> Cont<'a, This<M, B>, A>
where
    M: Monad<'a, A, B> + 'a + Clone,
    This<M, A>: Clone,
    A: Clone + 'a,
    B: Clone + 'a,
{
    Cont::cont(move |f| {
        let m = m.clone();
        M::bind(m, move |a| f(a))
    })
}

pub fn run<'a, A, M>(_: M, m: Cont<'a, This<M, A>, A>) -> This<M, A>
where
    M: Monad<'a, A, A> + 'a,
    A: Clone + 'a,
{
    Cont::run_cont(m, M::pure)
}

// Heavily borrowed from here: https://github.com/RustyYato/type-families
pub type This<T, A> = <T as Family<A>>::This;
pub trait Family<A> {
    type This;
}

pub trait Functor<'a, A, B>: Family<A> + Family<B> {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        A: 'a,
        B: 'a;
}

pub trait Pure<'a, A>: Family<A> {
    fn pure(value: A) -> This<Self, A>
    where
        A: 'a + Clone;
}

pub trait Applicative<'a, A, B>: Functor<'a, A, B> + Pure<'a, A> + Pure<'a, B> {
    fn apply<F>(a: This<Self, F>, b: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + 'a + Clone,
        Self: Applicative<'a, F, A>,
        A: 'a + Clone,
        B: 'a + Clone,
    {
        Self::lift_a2(move |q: F, r| q(r), a, b)
    }

    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + 'a + Clone,
        Self: Pure<'a, C>,
        Self: Family<C>,
        A: 'a + Clone,
        B: 'a + Clone,
        C: 'a + Clone;
}

pub trait Monad<'a, A, B>: Applicative<'a, A, B> {
    fn bind<F>(a: This<Self, A>, f: F) -> This<Self, B>
    where
        F: Fn(A) -> This<Self, B> + Clone + 'a;
    fn compose<F, G, C>(f: F, g: G, a: A) -> This<Self, C>
    where
        F: FnOnce(A) -> This<Self, B>,
        G: Fn(B) -> This<Self, C> + Clone + 'a,
        Self: Monad<'a, B, C>,
    {
        Self::bind(f(a), g)
    }
}

#[phantom]
#[derive(Clone, Copy)]
pub struct ContFamily<'a, R>;

impl<'a, R: 'a, A: 'a> Family<A> for ContFamily<'a, R> {
    type This = Cont<'a, R, A>;
}

impl<'a, R: 'a, A: 'a> Pure<'a, A> for ContFamily<'a, R> {
    fn pure(value: A) -> This<Self, A>
    where
        A: Clone + 'a,
    {
        Cont::pure(value)
    }
}

impl<'a, R: 'a, A: 'a, B: 'a> Functor<'a, A, B> for ContFamily<'a, R> {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + 'a + Clone,
    {
        this.fmap(f)
    }
}

impl<'a, R: 'a + Clone, A: 'a + Clone, B: 'a + Clone> Applicative<'a, A, B> for ContFamily<'a, R> {
    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + 'a + Clone,
        C: 'a + Clone,
    {
        a.bind(move |x: A| {
            let f = f.clone();
            let b = b.clone();
            b.bind(move |y: B| {
                let x = x.clone();
                Cont::pure(f(x, y))
            })
        })
    }
}

impl<'a, A, B, R> Monad<'a, A, B> for ContFamily<'a, R>
where
    R: Clone + 'a,
    A: Clone + 'a,
    B: Clone + 'a,
{
    fn bind<K>(a: This<Self, A>, k: K) -> This<Self, B>
    where
        K: Fn(A) -> Cont<'a, R, B> + 'a + Clone,
    {
        Cont::bind(a, k)
    }
}

//Heavily borrowed from the do_notation crate, but I feel like I need to rewrite it from
//scratch as it doesn't handle a lot of things that you want it to.
#[macro_export]
macro_rules! mdo {
  // return
  (pure $r:expr ;) => {
    $crate::Cont::pure($r)
  };

  // let-binding
  (let $p:pat = $e:expr ; $($r:tt)*) => {{
    let $p = $e;
    mdo!($($r)*)
  }};

  // const-bind
  (_ <- $x:expr ; $($r:tt)*) => {
    $crate::Cont::bind($x, move |_| { mdo!($($r)*) })
  };

  // const-bind
  (_ <- pure $p:expr ; $($r:tt)*) => {
    $crate::Cont::bind($crate::Cont::pure($p), move |_| { mdo!($($r)*) })
  };

  // bind
  ($binding:ident <- $x:expr ; $($r:tt)*) => {
    $crate::Cont::bind($x, move |$binding| { mdo!($($r)*) })
  };

  // bind
  ( ($binding:ident : $type:ty) <- $x:expr ; $($r:tt)*) => {
    $crate::Cont::bind($x, move |$binding:$type| { mdo!($($r)*) })
  };

  // bind
  ($binding:ident <- pure $x:expr ; $($r:tt)*) => {
    $crate::Cont::bind($crate::Cont::pure($x), move |$binding| { mdo!($($r)*) })
  };

  // const-bind
  ($e:expr ; $($a:tt)*) => {
    $crate::Cont::bind($e, move |_| mdo!($($a)*))
  };

  // const-bind
  (pure $e:expr ; $($a:tt)*) => {
    $crate::Cont::bind($crate::Cont::pure($e), move |_| mdo!($($a)*))
  };

  // pure
  ($a:expr) => {
    $a
  }
}

#[derive(Clone, Copy)]
pub struct OptionFamily;

impl<A> Family<A> for OptionFamily {
    type This = Option<A>;
}

impl<'a, A> Pure<'a, A> for OptionFamily {
    fn pure(value: A) -> This<Self, A> {
        Some(value)
    }
}

impl<'a, A, B> Functor<'a, A, B> for OptionFamily {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + Clone,
    {
        this.map(f)
    }
}

impl<'a, A, B> Applicative<'a, A, B> for OptionFamily {
    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C,
    {
        match (a, b) {
            (Some(x), Some(y)) => Some(f(x, y)),
            _ => None,
        }
    }
}

impl<'a, A, B> Monad<'a, A, B> for OptionFamily {
    fn bind<K>(a: This<Self, A>, k: K) -> This<Self, B>
    where
        K: Fn(A) -> Option<B>,
    {
        a.and_then(k)
    }
}

#[derive(Clone, Copy)]
pub struct VecFamily;

impl<A> Family<A> for VecFamily {
    type This = Vec<A>;
}

impl<'a, A> Pure<'a, A> for VecFamily {
    fn pure(value: A) -> This<Self, A> {
        vec![value]
    }
}

impl<'a, A, B> Functor<'a, A, B> for VecFamily {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + Clone,
    {
        this.into_iter().map(f).collect()
    }
}

impl<'a, A, B> Applicative<'a, A, B> for VecFamily {
    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + Clone,
        C: Clone,
        A: Clone,
        B: Clone,
    {
        a.into_iter()
            .flat_map(move |x| {
                let f = f.clone();
                let b = b.clone();
                b.into_iter().flat_map(move |y| vec![f(x.clone(), y)])
            })
            .collect()
    }
}

impl<'a, A, B> Monad<'a, A, B> for VecFamily {
    fn bind<K>(a: This<Self, A>, k: K) -> This<Self, B>
    where
        K: Fn(A) -> Vec<B>,
    {
        a.into_iter().flat_map(k).collect()
    }
}

#[derive(Clone, Copy)]
pub struct IdentityFamily;

#[derive(Clone, Copy, Debug)]
pub struct Identity<A>(A);

impl<A: std::fmt::Debug> std::fmt::Display for Identity<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.0)
    }
}

impl<A> Family<A> for IdentityFamily {
    type This = Identity<A>;
}

impl<'a, A> Pure<'a, A> for IdentityFamily {
    fn pure(value: A) -> This<Self, A> {
        Identity(value)
    }
}

impl<'a, A, B> Functor<'a, A, B> for IdentityFamily {
    fn map<F>(f: F, this: This<Self, A>) -> This<Self, B>
    where
        F: Fn(A) -> B + Clone,
    {
        Identity(f(this.0))
    }
}

impl<'a, A, B> Applicative<'a, A, B> for IdentityFamily {
    fn lift_a2<C, F>(f: F, a: This<Self, A>, b: This<Self, B>) -> This<Self, C>
    where
        F: Fn(A, B) -> C + Clone,
        C: Clone,
        A: Clone,
        B: Clone,
    {
        Identity(f(a.0, b.0))
    }
}

impl<'a, A, B> Monad<'a, A, B> for IdentityFamily {
    fn bind<K>(a: This<Self, A>, k: K) -> This<Self, B>
    where
        K: Fn(A) -> Identity<B>,
    {
        k(a.0)
    }
}

pub trait TyImp {
    type Imp;
}
pub trait TyLink<A> {
    type Related;
}
type R<A, B> = <A as TyLink<B>>::Related;

pub struct PureI<F: TyImp> {
    pub pure: fn(value: F::Imp) -> F,
}

impl<A> TyImp for Identity<A> {
    type Imp = A;
}
impl<A, B> TyLink<B> for Identity<A> {
    type Related = Identity<B>;
}

pub fn pure_i<A>(a: A) -> Identity<A> {
    Identity(a)
}

pub fn pure_identity<A>() -> PureI<Identity<A>> {
    PureI {
        pure: pure_i::<<Identity<A> as TyImp>::Imp>,
    }
}

pub struct FunctorI<A, B, M>
where
    M: TyLink<A> + TyLink<B>,
{
    pub map: fn(R<M, A>, f: &dyn Fn(A) -> B) -> R<M, B>,
}
pub fn map_i<A, B>(a: Identity<A>, f: &dyn Fn(A) -> B) -> Identity<B> {
    Identity(f(a.0))
}

pub fn functor_identity<A, B>() -> FunctorI<A, B, Identity<A>> {
    FunctorI { map: map_i::<A, B> }
}

pub fn lift_a2_i<A, B, C>(f: &dyn Fn(A, B) -> C, a: Identity<A>, b: Identity<B>) -> Identity<C> {
    Identity(f(a.0, b.0))
}
pub fn apply_i<A, B>(a: Identity<&dyn Fn(A) -> B>, b: Identity<A>) -> Identity<B> {
    Identity(a.0(b.0))
}
pub struct ApplicativeI<'a, A: 'a, B: 'a, C, M>
where
    M: TyLink<A> + TyLink<B> + TyLink<C> + TyLink<&'a dyn Fn(A) -> B>,
{
    pub lift_a2: fn(f: &dyn Fn(A, B) -> C, a: R<M, A>, b: R<M, B>) -> R<M, C>,
    pub apply: fn(a: R<M, &'a dyn Fn(A) -> B>, b: R<M, A>) -> R<M, B>,
}
pub fn applicative_identity<'a, A, B, C>() -> ApplicativeI<'a, A, B, C, Identity<A>> {
    ApplicativeI {
        lift_a2: lift_a2_i::<A, B, C>,
        apply: apply_i::<A, B>,
    }
}

pub fn bind_i<A, B>(a: Identity<A>, f: &dyn Fn(A) -> Identity<B>) -> Identity<B> {
    f(a.0)
}
type BindT<'a, A, B, M> = fn(a: R<M, A>, f: &dyn Fn(A) -> R<M, B>) -> R<M, B>;
type PureT<'a, A, M> = fn(a: A) -> R<M, A>;
type MapT<'a, A, B, M> = fn(a: R<M, A>, f: &dyn Fn(A) -> B) -> R<M, B>;
use std::any::Any;
pub struct MonadI<M> {
    // These are function pointers and we ought to cast them carefully.
    bind_p: Box<dyn Any>,
    pure_p: Box<dyn Any>,
    map_p: Box<dyn Any>,
    m: std::marker::PhantomData<M>,
}
impl<M> MonadI<M> {
    pub fn new<A, B>(
        bind: Box<BindT<A, B, M>>,
        pure: Box<PureT<A, M>>,
        map: Box<MapT<A, B, M>>,
    ) -> Self
    where
        M: TyLink<A> + TyLink<B>,
        R<M, A>: 'static,
        R<M, B>: 'static,
        A: 'static,
        B: 'static,
    {
        MonadI {
            bind_p: {
                let b = bind as Box<dyn Any>;
                assert!(b.is::<BindT<A, B, M>>());
                b
            },
            pure_p: {
                let p = pure as Box<dyn Any>;
                assert!(p.is::<PureT<A, M>>());
                p
            },
            map_p: {
                let m = map as Box<dyn Any>;
                assert!(m.is::<MapT<A, B, M>>());
                m
            },
            m: std::marker::PhantomData,
        }
    }
    pub fn bind<A, B>(&self, a: R<M, A>, f: &dyn Fn(A) -> R<M, B>) -> R<M, B>
    where
        M: TyLink<A> + TyLink<B>,
        A: 'static,
        R<M, B>: 'static,
        R<M, A>: 'static,
    {
        let bind = self.bind_p.downcast_ref::<BindT<A, B, M>>().unwrap();
        bind(a, f)
    }

    pub fn pure<A>(&self, a: A) -> R<M, A>
    where
        M: TyLink<A>,
        R<M, A>: 'static,
        A: 'static,
    {
        let pure = self.pure_p.downcast_ref::<&fn(a: A) -> R<M, A>>().unwrap();
        pure(a)
    }

    pub fn map<A, B>(&self, a: R<M, A>, f: &dyn Fn(A) -> B) -> R<M, B>
    where
        M: TyLink<A> + TyLink<B>,
        R<M, B>: 'static,
        R<M, A>: 'static,
        A: 'static,
        B: 'static,
    {
        let map = self
            .map_p
            .downcast_ref::<&fn(a: R<M, A>, f: &dyn Fn(A) -> B) -> R<M, B>>()
            .unwrap();
        map(a, f)
    }
}
pub fn monad_identity<A: 'static, B: 'static>() -> MonadI<Identity<A>> {
    MonadI::new(
        Box::new(bind_i::<A, B>),
        Box::new(pure_i::<A>),
        Box::new(map_i::<A, B>),
    )
}
// i :: Monad m => m a -> Cont (m b) a
pub fn lift_i<'a, A, B, M>(m: MonadI<M>, ma: R<M, A>) -> Cont<'a, R<M, B>, A>
where
    M: TyLink<A> + TyLink<B> + 'a + Clone,
    R<M, A>: 'static + Clone,
    R<M, B>: 'static,
    A: Clone + 'static,
    B: Clone + 'a,
{
    Cont::cont(move |f| m.bind::<A, B>(ma.clone(), &*f))
}

pub fn lift_ii<'a, A, B>(ma: Identity<A>) -> Cont<'a, Identity<B>, A>
where
    A: Clone + 'static,
    B: Clone + 'static,
{
    lift_i::<A, B, Identity<A>>(monad_identity::<A, B>(), ma)
}
